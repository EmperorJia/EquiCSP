import itertools
import numpy as np
import torch
import hydra

from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from hydra.experimental import compose
from hydra import initialize_config_dir
from pathlib import Path

import smact
from smact.screening import pauling_test

import sys
sys.path.append('.')

from equicsp.common.constants import CompScalerMeans, CompScalerStds
from equicsp.common.data_utils import StandardScaler, chemical_symbols
from equicsp.pl_data.dataset import TensorCrystDataset
from equicsp.pl_data.datamodule import worker_init_fn

from torch_geometric.data import DataLoader

CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.)

import pdb

recommand_step_lr = {
    'csp':{
        "perov_5": 5e-7,
        "carbon_24": 5e-6,
        "mp_20": 5e-6,
        "mpts_52": 1e-5
    },
    'csp_multi':{
        "perov_5": 5e-7,
        "carbon_24": 5e-7,
        "mp_20": 5e-6,
        "mpts_52": 1e-5
    },
    'gen':{
        "perov_5": 1e-6,
        "carbon_24": 1e-5,
        "mp_20": 5e-6
    },
}

def lattices_to_params_shape(lattices):

    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) /
                            (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles

def load_data(file_path):
    if file_path[-3:] == 'npy':
        data = np.load(file_path, allow_pickle=True).item()
        for k, v in data.items():
            if k == 'input_data_batch':
                for k1, v1 in data[k].items():
                    data[k][k1] = torch.from_numpy(v1)
            else:
                data[k] = torch.from_numpy(v).unsqueeze(0)
    else:
        data = torch.load(file_path, map_location='cpu')
    return data


def get_model_path(eval_model_name):
    import equicsp
    model_path = (
        Path(equicsp.__file__).parent / 'prop_models' / eval_model_name)
    return model_path


def load_config(model_path):
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
    return cfg


def load_model(model_path, load_data=False, testing=True, last_ckpt = True, load_scaler = False):
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
        model = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )
        ckpts = list(model_path.glob('*.ckpt'))
        if len(ckpts) > 0:
            # ckpt_epochs = np.array(
            #     [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts if 'last' not in ckpt.parts[-1]])
            # ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
            # if 'last.ckpt' in [_.parts[-1] for _ in ckpts]:
            if last_ckpt:
                ckpt = str(model_path) + '/last.ckpt'
            else:
                my_ckpts = [x for x in ckpts if x.parts[-1] != 'last.ckpt']
                ckpts = my_ckpts
                print(ckpts)
                ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
                ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        model.load_state_dict(torch.load(ckpt)['state_dict'], strict=False)
        # model = model.load_from_checkpoint(ckpt, strict=False)
        if load_scaler:
            model.lattice_scaler = torch.load(model_path / 'lattice_scaler.pt')
            model.scaler = torch.load(model_path / 'prop_scaler.pt')


        if load_data:
            datamodule = hydra.utils.instantiate(
                cfg.data.datamodule, _recursive_=False, scaler_path=None
            )
            if testing:
                datamodule.setup('test')
                test_loader = datamodule.test_dataloader()[0]
            else:
                datamodule.setup()
                train_loader = datamodule.train_dataloader(shuffle=False)
                val_loader = datamodule.val_dataloader()[0]
                test_loader = (train_loader, val_loader)
        else:
            test_loader = None

    return model, test_loader, cfg

def load_dataset(data_path, testing=True):
    with initialize_config_dir(str(data_path)):
        cfg = compose(config_name='hparams')
        datamodule = hydra.utils.instantiate(
            cfg.data.datamodule, _recursive_=False, scaler_path=None
        )
        if testing:
            datamodule.setup('test')
            test_loader = datamodule.test_dataloader()[0]
        else:
            datamodule.setup()
            train_loader = datamodule.train_dataloader(shuffle=False)
            val_loader = datamodule.val_dataloader()[0]
            test_loader = (train_loader, val_loader)
        
    return test_loader

def load_datasets(data_path, num_dataset=1):
    with initialize_config_dir(str(data_path)):
        cfg = compose(config_name='hparams')
        loaders = []
        sum = 0
        print('Gernerating dataloaders: ')
        for i in range(num_dataset):
            test_dataset = [ hydra.utils.instantiate(dataset_cfg, _target_='equicsp.pl_data.dataset.DisCrystDataset', rank=i, num_dis=num_dataset) for dataset_cfg in cfg.data.datamodule.datasets.test ]
            test_loader = DataLoader(
                test_dataset[0],
                shuffle=False,
                batch_size=cfg.data.datamodule.batch_size.test,
            )
            loaders.append(test_loader)
            sum += len(test_dataset[0])
            print(f'generated {sum} data')
        
    return loaders

def get_crystals_list(
        frac_coords, atom_types, lengths, angles, num_atoms):
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
    """
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_array_list.append({
            'frac_coords': cur_frac_coords.detach().cpu().numpy(),
            'atom_types': cur_atom_types.detach().cpu().numpy(),
            'lengths': cur_lengths.detach().cpu().numpy(),
            'angles': cur_angles.detach().cpu().numpy(),
        })
        start_idx = start_idx + num_atom
    return crystal_array_list


def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    # if len(list(itertools.product(*ox_combos))) > 1e5:
    #     return False
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1 or max(crystal.lattice.abc) > 40:
        return False
    else:
        return True


def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()


def prop_model_eval(eval_model_name, crystal_array_list):

    model_path = get_model_path(eval_model_name)

    model, _, _ = load_model(model_path, last_ckpt = False, load_scaler = True)
    cfg = load_config(model_path)

    dataset = TensorCrystDataset(
        crystal_array_list, cfg.data.niggli, cfg.data.primitive,
        cfg.data.graph_method, cfg.data.preprocess_workers,
        cfg.data.lattice_scale_method)

    dataset.scaler = model.scaler.copy()

    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=256,
        num_workers=0,
        worker_init_fn=worker_init_fn)

    model.eval()

    all_preds = []

    for batch in loader:
        preds = model(batch)
        model.scaler.match_device(preds)
        scaled_preds = model.scaler.inverse_transform(preds)
        all_preds.append(scaled_preds.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).squeeze(1)
    return all_preds.tolist()


def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


def compute_cov(crys, gt_crys,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):
    struc_fps = [c.struct_fp for c in crys]
    comp_fps = [c.comp_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]
    gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)

    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        'amsd_recall': np.mean(struc_recall_dist),
        'amsd_precision': np.mean(struc_precision_dist),
        'amcd_recall': np.mean(comp_recall_dist),
        'amcd_precision': np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
        'comp_recall_dist': comp_recall_dist.tolist(),
        'comp_precision_dist': comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict
