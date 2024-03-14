
# Copyright 2023 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict
from copy import deepcopy

from torch.utils.data import ConcatDataset

from efm_datasets.dataloaders.utils.transforms import no_transform
from efm_datasets.utils.config import load_class, cfg_has, cfg_add_to_dict, get_from_cfg_list, merge_dict, to_namespace
from efm_datasets.utils.data import flatten, keys_in
from efm_datasets.utils.types import is_namespace


def setup_dataset(cfg, root='efm_datasets/dataloaders'):
    """Create a dataset from configuration"""

    shared_keys = ['context', 'labels', 'labels_context']

    num_datasets = 0
    for key, val in cfg.__dict__.items():
        if key not in shared_keys and not is_namespace(val):
            num_datasets = max(num_datasets, len(val))

    datasets = []
    for i in range(num_datasets):
        args = {}
        for key, val in cfg.__dict__.items():
            if not is_namespace(val):
                cfg_add_to_dict(args, cfg, key, i if key not in shared_keys else None)
        args['data_transform'] = no_transform
        name = get_from_cfg_list(cfg, 'name', i)
        repeat = get_from_cfg_list(cfg, 'repeat', i)
        dataset = load_class(name + 'Dataset', root)(**args)
        if cfg_has(cfg, 'repeat') and repeat > 1:
            dataset = ConcatDataset([dataset for _ in range(repeat)])
        datasets.append(dataset)

    return datasets


def setup_datasets(cfg, concat_modes=('train', 'mixed'), stack=True):
    """Create multiple datasets from configuration"""

    datasets_cfg = {}
    for key in cfg.__dict__.keys():
        datasets_cfg[key] = cfg.__dict__[key]
        for mode in ['train', 'validation']:
            if key.startswith(mode) and key != mode and mode in cfg.__dict__.keys():
                datasets_cfg[key] = to_namespace(merge_dict(deepcopy(
                    cfg.__dict__[mode].__dict__), cfg.__dict__[key].__dict__))

    datasets = {}
    for key, val in list(datasets_cfg.items()):
        if 'name' in val.__dict__.keys():
            datasets[key] = setup_dataset(val)
            datasets_cfg[key] = [datasets_cfg[key]] * len(datasets[key])
            for mode in concat_modes:
                if key.startswith(mode) and len(datasets[key]) > 1:
                    datasets[key] = ConcatDataset(datasets[key])
        else:
            datasets_cfg.pop(key)

    if stack:
        datasets = stack_datasets(datasets)

    modes = ['train', 'mixed', 'validation', 'test']
    reduced_datasets_cfg = {key: [] for key in modes}
    for key, val in datasets_cfg.items():
        for mode in modes:
            if key.startswith(mode):
                reduced_datasets_cfg[mode].append(val)
    for key in list(reduced_datasets_cfg.keys()):
        reduced_datasets_cfg[key] = flatten(reduced_datasets_cfg[key])
        if len(reduced_datasets_cfg[key]) == 0:
            reduced_datasets_cfg.pop(key)
    datasets_cfg = reduced_datasets_cfg

    if 'train' in datasets_cfg:
        datasets_cfg['train'] = datasets_cfg['train'][0]

    return datasets, datasets_cfg


def reduce(data, modes, train_modes):
    """Reduce dataset dictionary to a single value per mode"""
    reduced = {
        mode: flatten([val for key, val in data.items() if mode in key])
        for mode in modes
    }
    for key, val in list(reduced.items()):
        if len(val) == 0:
            reduced.pop(key)
    for mode in keys_in(reduced, train_modes):
        reduced[mode] = reduced[mode][0]
    return reduced


def stack_datasets(datasets):
    """Stack datasets together to create a larger dataset"""

    all_modes = ['train', 'mixed', 'validation', 'test']
    train_modes = ['train', 'mixed']

    stacked_datasets = OrderedDict()

    for mode in all_modes:
        stacked_datasets[mode] = []
        for key, val in datasets.items():
            if mode in key:
                stacked_datasets[mode].append(val)
        stacked_datasets[mode] = flatten(stacked_datasets[mode])

    for mode in train_modes:
        length = len(stacked_datasets[mode])
        if length == 1:
            stacked_datasets[mode] = stacked_datasets[mode][0]
        elif length > 1:
            stacked_datasets[mode] = ConcatDataset(stacked_datasets[mode])
        for key in list(datasets.keys()):
            if key.startswith(mode) and key != mode:
                datasets.pop(key)

    return stacked_datasets
