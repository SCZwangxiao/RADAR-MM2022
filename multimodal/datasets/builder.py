import platform
import random
from distutils.version import LooseVersion
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

from .samplers import ClassSpecificDistributedSampler, DistributedSampler, KwaiNodeCollator

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
BLENDINGS = Registry('blending')


def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict | None, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset,
                     videos_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     persistent_workers=True,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (:obj:`Dataset`): A PyTorch dataset.
        videos_per_gpu (int): Number of videos on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data
            loading for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed
            training. Default: 1.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0.
            Default: True
        kwargs (dict, optional): Any keyword argument to be used to initialize
            DataLoader.

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    dataset_type = kwargs.pop('dataset_type', 'normal')
    if dataset_type == 'normal':
        pass
    elif dataset_type == 'graph':
        return build_graph_dataloader(dataset,
                                      videos_per_gpu,
                                      workers_per_gpu,
                                      num_gpus,
                                      dist,
                                      shuffle,
                                      seed,
                                      drop_last,
                                      pin_memory,
                                      **kwargs)
    else:
        raise NotImplementedError
    rank, world_size = get_dist_info()
    sample_by_class = getattr(dataset, 'sample_by_class', False)

    if dist:
        if sample_by_class:
            dynamic_length = getattr(dataset, 'dynamic_length', True)
            sampler = ClassSpecificDistributedSampler(
                dataset,
                world_size,
                rank,
                dynamic_length=dynamic_length,
                shuffle=shuffle,
                seed=seed)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=shuffle, seed=seed)
        shuffle = False
        batch_size = videos_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * videos_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if LooseVersion(torch.__version__) >= LooseVersion('1.7.0'):
        kwargs['persistent_workers'] = persistent_workers

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=videos_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader


def build_graph_dataloader(dataset,
                           videos_per_gpu,
                           workers_per_gpu,
                           num_gpus,
                           dist,
                           shuffle,
                           seed,
                           drop_last,
                           pin_memory,
                           collator,
                           sampler):
    import dgl
    if num_gpus != 1:
        raise NotImplementedError
    if seed:
        if workers_per_gpu > 1:
            print('Warning, seed for dgl sampler is only when num workers <= 1, https://github.com/dmlc/dgl/issues/1110')
        dgl.seed(seed)
    
    rank, world_size = get_dist_info()
    g = dataset.g
    sampler_fn = getattr(dgl.dataloading, sampler.pop('type'))(**sampler)
    collator_module = globals()[collator.pop('type')](g=g,
                                                      block_sampler=sampler_fn,
                                                      test_mode=dataset.test_mode,
                                                      **collator)
    batch_size = num_gpus * videos_per_gpu
    num_workers = num_gpus * workers_per_gpu
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
    
    data_loader = DataLoader(
        dataset,
        batch_size=videos_per_gpu,
        collate_fn=collator_module.collate,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=workers_per_gpu)

    return data_loader

def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)