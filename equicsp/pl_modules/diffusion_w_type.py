import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter

from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
import random

from equicsp.common.utils import PROJECT_ROOT
from equicsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from equicsp.pl_modules.diff_utils import d_log_p_wrapped_normal

MAX_ATOMIC_NUM=100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}

# Directional statistics（Mardia)
    
def calc_mean_sin_cos(data_tensor):
    m_sin = torch.sin(data_tensor).mean(dim=1)  # Mean of sine values along dim=1
    m_cos = torch.cos(data_tensor).mean(dim=1)  # Mean of cosine values along dim=1
    return m_sin, m_cos

def calc_grouped_angles_mean_in_radians(data_tensor, groups):
    # 将[0,1]映射为圆上的弧度
    data_tensor = data_tensor * 2*math.pi

    # 初始化一个 tensor 用于保存分组后的数据
    data_tensor_sin = torch.sin(data_tensor)
    data_tensor_cos = torch.cos(data_tensor)

    # 使用 scatter 计算每个组的累加和
    sum_sin = scatter(data_tensor_sin, groups, dim=0, reduce='sum')
    sum_cos = scatter(data_tensor_cos, groups, dim=0, reduce='sum')

    # 计算每个组的数量
    group_counts = scatter(torch.ones_like(data_tensor), groups, dim=0, reduce='sum')

    # 计算分组的平均值
    mean_sin = sum_sin / group_counts
    mean_cos = sum_cos / group_counts

    mean_angle = torch.atan2(mean_sin, mean_cos)  # Calculate mean angle in radians using atan2 for stability

    # Adjust mean_angle to be in the range [0, 2*pi)
    mean_angle = torch.where(mean_angle >= 0, mean_angle, mean_angle + 2 * math.pi)

    # 圆上的弧度映射回[0, 1]
    mean_angle = mean_angle / (2*math.pi)

    return mean_angle

def d_log_x(score, d_mean_x, groups, num_atoms):
    score_sum = scatter(score, groups, dim=0, reduce='sum')
    score_sum = score_sum.repeat_interleave(num_atoms, dim=0)
    score_mean = score_sum * (-d_mean_x)
    result = score_mean + score
    return result


def d_mean_angle_x(data_tensor, groups, num_atoms):
    # 将[0,1]映射为圆上的弧度
    data_tensor = data_tensor * 2*math.pi

    # 初始化一个 tensor 用于保存分组后的数据
    data_tensor_sin = torch.sin(data_tensor)
    data_tensor_cos = torch.cos(data_tensor)

    # 使用 scatter 计算每个组的累加和
    u = scatter(data_tensor_sin, groups, dim=0, reduce='mean')
    v = scatter(data_tensor_cos, groups, dim=0, reduce='mean')

    n = scatter(torch.ones_like(data_tensor), groups, dim=0, reduce='sum')

    u = u.repeat_interleave(num_atoms, dim=0)
    v = v.repeat_interleave(num_atoms, dim=0)
    n = n.repeat_interleave(num_atoms, dim=0)

    result = (v*data_tensor_cos + u*data_tensor_sin) / (u**2 + v**2) / n

    return result

### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)
        # add type
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, pred_type = True, smooth = True)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        # add proj
        self.use_proj = self.hparams.use_proj

        if hasattr(self.hparams, "dev_norm"):
            self.dev_norm = self.hparams.dev_norm
        else:
            self.dev_norm = False

        self.trans_matrix = torch.tensor([
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ],
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ],
            [
                [0, 1, 0],
                [0, 0, 1], 
                [1, 0, 0]
            ],
            [
                [0, 0, 1], 
                [1, 0, 0], 
                [0, 1, 0]
            ],
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ]
        ], dtype=float, device='cuda')


    def get_replace_lattice(self, indices, input_lattice):

        # 选择对应的置换矩阵
        selected_matrices = self.trans_matrix[indices]

        # 获取前三维和后三维数据
        input_lattice_front = input_lattice[:, :3]
        input_lattice_back = input_lattice[:, 3:]

        # 对前三维进行置换矩阵乘法
        # print('数据类型：',  input_lattice_back.dtype)
        result_front = torch.bmm(input_lattice_front.unsqueeze(1), selected_matrices.float())
        result_front = result_front.squeeze(1)

        
        # 对后三维进行置换矩阵乘法
        result_back = torch.bmm(input_lattice_back.unsqueeze(1), selected_matrices.float())
        result_back = result_back.squeeze(1)

        # 合并前三维和后三维结果
        result = torch.cat((result_front, result_back), dim=1)

        return result
    
    def get_replace_coord(self, indices, batch, coord):

        # 选择对应的置换矩阵
        selected_matrices = self.trans_matrix[indices]

        #print('batch.shape: ', batch.shape)
        #print('selected_matrices.shape: ', selected_matrices.shape)
        selected_matrices = selected_matrices.repeat_interleave(batch, dim=0)

        # 对前三维进行置换矩阵乘法
        result = torch.bmm(coord.unsqueeze(1), selected_matrices.float())
        result = result.squeeze(1)

        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        """
        encode crystal structures to latents.
        """
        hidden = self.encoder(batch)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z


    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]

        sample_norm = self.sigma_scheduler.sample_norm[batch.num_atoms-1, times]
        sample_norm_per_atom = sample_norm.repeat_interleave(batch.num_atoms)[:, None]

        rad_angles = torch.deg2rad(batch.angles)

        if self.use_proj:
            lengths = torch.log(batch.lengths)
            angles = torch.tan(rad_angles - math.pi / 2)
        else:
            lengths, angles = batch.lengths, rad_angles
        lattices = torch.cat([lengths, angles], dim=1)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None] * lattices + c1[:, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        # add type
        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
        rand_t = torch.randn_like(gt_atom_types_onehot)
        atom_type_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)


        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        # add type
        pred_l, pred_x, pred_score, pred_t = self.decoder(time_emb, atom_type_probs, input_frac_coords, 
                                      input_lattice, batch.num_atoms, batch.batch)
        
        indices = [random.randint(0, 4) for _ in range(batch_size)]
        input_lattice_replace = self.get_replace_lattice(indices, input_lattice)
        input_frac_coords_replace = self.get_replace_coord(indices, batch.num_atoms, input_frac_coords)
        # add type
        pred_l_re, pred_x_re, pred_score_re, pred_t_re = self.decoder(time_emb, atom_type_probs, input_frac_coords_replace, 
                                      input_lattice_replace, batch.num_atoms, batch.batch)
                                      
        loss_replace_l = F.mse_loss(pred_l_re, self.get_replace_lattice(indices, pred_l))
        loss_replace_f = F.mse_loss(pred_x_re, self.get_replace_coord(indices, batch.num_atoms, pred_x)) + F.mse_loss(pred_score_re, self.get_replace_coord(indices, batch.num_atoms, pred_score))
        # add type
        loss_replace_t = F.mse_loss(pred_t_re, pred_t)



        invarance_reference = calc_grouped_angles_mean_in_radians((sigmas_per_atom * rand_x)%1.0, batch.batch)
        invarance_reference = invarance_reference.repeat_interleave(batch.num_atoms, dim=0)
        invariance_noise = (sigmas_per_atom * rand_x - invarance_reference)%1.


        kappa = self.sigma_scheduler.kappa_matrix[batch.num_atoms, times]
        kappa_norm = self.sigma_scheduler.kappa_norm[batch.num_atoms, times]
        kappa_per_atom = kappa.repeat_interleave(batch.num_atoms)[:, None]
        kappa_norm_per_atom  = kappa_norm.repeat_interleave(batch.num_atoms)[:, None]

        tar_score = kappa_per_atom*torch.sin(invariance_noise*2*math.pi)*2*math.pi
        tar_score = tar_score / torch.sqrt(kappa_norm_per_atom)
        
        tar_x = torch.where(invariance_noise<=0.5, invariance_noise, invariance_noise-1.0) / torch.sqrt(sample_norm_per_atom)

        loss_mean=0

        


        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x) 
        loss_score = F.mse_loss(pred_score, tar_score) 
        # add type
        loss_type = F.mse_loss(pred_t, rand_t)

        loss_lat_coord = 0

        loss = (
            loss_lattice +
            self.hparams.cost_coord * loss_coord + 
            loss_replace_l + 
            loss_replace_f + 
            # add type
            self.hparams.cost_type * loss_type + 
            self.hparams.cost_replace_type * loss_replace_t + 
            loss_score
        )

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord, 
            'loss_replace_l' : loss_replace_l,
            'loss_replace_f' : loss_replace_f, 
            'loss_lat_coord': loss_lat_coord,
            'loss_score': loss_score, 
            # add type
            'loss_type': loss_type,
            'loss_replace_t': loss_replace_t,
            'loss_mean': loss_mean,
        }

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5):

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 6]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        # add type
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)


        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            # add type
            'atom_types': t_T,
            'frac_coords' : x_T % 1.,
            'lattices' : lattice_params_to_matrix_torch(l_T[:, 0:3], torch.rad2deg(l_T[:, 3:6])),
            'diff_lattices' : l_T, 
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]

            sigma_norm = self.sigma_scheduler.kappa_norm[batch.num_atoms, t]
            sigmas_norm_per_atom = sigma_norm.repeat_interleave(batch.num_atoms)[:, None]

            sample_norm = self.sigma_scheduler.sample_norm[batch.num_atoms-1, t]
            sample_norm_per_atom = sample_norm.repeat_interleave(batch.num_atoms)[:, None]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['diff_lattices']
            # add type
            t_t = traj[t]['atom_types']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            # add type
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)


            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            # add type
            pred_l, pred_x, pred_score, pred_t = self.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch)

            pred_score = pred_score * torch.sqrt(sigmas_norm_per_atom)

            pred_x_inc = pred_x * torch.sqrt(sample_norm_per_atom)
            pred_x_sign = pred_x_inc.clip(-0.5, 0.5)
            d_mean_angle = d_mean_angle_x(pred_x_sign%1., batch.batch, batch.num_atoms)
            pred_x_d_log = d_log_x(pred_score, d_mean_angle, batch.batch, batch.num_atoms)

            x_t_minus_05 = x_t - step_size * pred_x_d_log + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # add type
            t_t_minus_05 = t_t

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            # add type
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            # add type
            pred_l, pred_x, pred_score, pred_t = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            pred_score = pred_score * torch.sqrt(sigmas_norm_per_atom)

            pred_x_inc = pred_x * torch.sqrt(sample_norm_per_atom)
            pred_x_sign = pred_x_inc.clip(-0.5, 0.5)
            d_mean_angle = d_mean_angle_x(pred_x_sign%1., batch.batch, batch.num_atoms)
            pred_x_d_log = d_log_x(pred_score, d_mean_angle, batch.batch, batch.num_atoms)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x_d_log + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t

            # add type
            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            # 弧度制转换
            if self.use_proj:
                lat_matrix = lattice_params_to_matrix_torch(torch.exp(l_t_minus_1[:, 0:3]), torch.rad2deg(torch.arctan(l_t_minus_1[:, 3:6]) + math.pi / 2))
            else:
                lat_matrix = lattice_params_to_matrix_torch(l_t_minus_1[:, 0:3], torch.rad2deg(l_t_minus_1[:, 3:6]))

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                # add type
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'diff_lattices' : l_t_minus_1, 
                'lattices' : lat_matrix              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            # add type
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_replace_l = output_dict['loss_replace_l']
        loss_replace_f = output_dict['loss_replace_f']
        loss_lat_coord = output_dict['loss_lat_coord']
        loss = output_dict['loss']
        loss_score = output_dict['loss_score']
        loss_mean = output_dict['loss_mean']
        # add type
        loss_type = output_dict['loss_type']
        loss_replace_t = output_dict['loss_replace_t']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord,
            'loss_replace_l': loss_replace_l, 
            'loss_replace_f': loss_replace_f, 
            # add type
            'loss_type': loss_type,
            'loss_replace_t': loss_replace_t,
            'loss_score': loss_score
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_score = output_dict['loss_score']
        # add type
        loss_type = output_dict['loss_type']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_loss_score': loss_score,
            # add type
            f'{prefix}_loss_type': loss_type,
        }

        return log_dict, loss


