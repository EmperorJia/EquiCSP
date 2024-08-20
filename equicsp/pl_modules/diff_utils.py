import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
from scipy.stats import norm
from abc import ABC, abstractmethod
from pathlib import Path
import equicsp

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
    return p_

def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (x + T * i) / sigma ** 2 * torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
    return p_ / p_wrapped_normal(x, sigma, N, T)

def sigma_norm(sigma, T=1.0, sn = 10000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T = T)
    return (normal_ ** 2).mean(dim = 0)

# normalization operation for better training in practice like most diffution model, which not mentioned in our paper
def sample_norm(sigma, T=1.0, sn = 10000, num_atoms=52):
    # created by 'equicsp/pl_modules/von_mises_norm.py'
    # for normalization
    
    sample_norm = torch.load(Path(equicsp.__file__).parent / 'normalization' / 'sample_norm.pth').to('cpu')
    return sample_norm

# normalization operation for better training in practice like most diffution model, which not mentioned in our paper
def kappa_func():
    # created by 'equicsp/pl_modules/von_mises_norm.py'
    # for normalization
    kappa_matrix = torch.load(Path(equicsp.__file__).parent / 'normalization' / 'kappa_matrix.pth')
    kappa_norm = torch.load(Path(equicsp.__file__).parent / 'normalization' / 'kappa_norm.pth')
    kappa_matrix = torch.tensor(kappa_matrix).float()
    kappa_norm = torch.tensor(kappa_norm).float()
    kappa_norm[1] = kappa_norm[1] + 1e-10
    return kappa_matrix, kappa_norm


class BetaScheduler(nn.Module):

    def __init__(
        self,
        timesteps,
        scheduler_mode,
        beta_start = 0.0001,
        beta_end = 0.02
    ):
        super(BetaScheduler, self).__init__()
        self.timesteps = timesteps
        if scheduler_mode == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif scheduler_mode == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'quadratic':
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)


        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        sigmas = torch.zeros_like(betas)

        sigmas[1:] = betas[1:] * (1. - alphas_cumprod[:-1]) / (1. - alphas_cumprod[1:])

        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sigmas', sigmas)

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)
    
    def uniform_sample_t_no_zero(self, batch_size, device):
        ts = np.random.choice(np.arange(2, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)

class SigmaScheduler(nn.Module):

    def __init__(
        self,
        timesteps,
        sigma_begin = 0.01,
        sigma_end = 1.0
    ):
        super(SigmaScheduler, self).__init__()
        self.timesteps = timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        sigmas = torch.FloatTensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps)))

        sample_norm_ = sample_norm(sigmas)
        self.register_buffer('sample_norm', torch.cat([torch.ones([52, 1]), sample_norm_], dim=1))

        kappa_matrix_, kappa_norm_ = kappa_func()
        self.register_buffer('kappa_matrix', torch.cat([torch.ones([53, 1]), kappa_matrix_], dim=1))
        self.register_buffer('kappa_norm', torch.cat([torch.ones([53, 1]), kappa_norm_], dim=1))
        
        self.register_buffer('sigmas', torch.cat([torch.zeros([1]), sigmas], dim=0))
        

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)

class LogNormalSampler:
    def __init__(self, p_mean=-1.2, p_std=1.2, even=False):
        self.p_mean = p_mean
        self.p_std = p_std

    def sample(self, bs, device):
        log_sigmas = self.p_mean + self.p_std * torch.randn(bs, device=device)
        sigmas = torch.exp(log_sigmas)
        weights = torch.ones_like(sigmas)
        return sigmas, weights
    
class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights
    
class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        # print(targ.device)
        # print(src.device)
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

class DummyGenerator:
    def randn(self, *args, **kwargs):
        return torch.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return torch.randint(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        return torch.randn_like(*args, **kwargs)
