# --------------------------------------------------------
# Group R-CNN
# Copyright (c) OpenMMLab. All rights reserved.
# Written by Shilong Zhang
# --------------------------------------------------------
from collections.abc import Mapping, Sequence
from functools import partial

import torch
import torch.nn.functional as F
from mmcv.parallel import collate
from mmcv.parallel.data_container import DataContainer
from mmcv.runner import get_dist_info
from mmdet.datasets.builder import worker_init_fn
from mmdet.datasets.samplers import (DistributedGroupSampler,
                                     DistributedSampler, GroupSampler)
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def multi_collate_fn(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size. This is mainly used in query_support dataloader. The main
    difference with the :func:`collate_fn`  in mmcv is it can process
    list[list[DataContainer]].

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data.
    2. cpu_only = False, stack = True, e.g., images tensors.
    3. cpu_only = False, stack = False, e.g., gt bboxes.

    Args:
        batch (list[list[:obj:`mmcv.parallel.DataContainer`]] |
            list[:obj:`mmcv.parallel.DataContainer`]): Data of
            single batch.
        samples_per_gpu (int): The number of samples of single GPU.
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    # This is usually a case in query_support dataloader, which
    # the :func:`__getitem__` of dataset return more than one images.
    # Here we process the support batch data in type of
    # List: [ List: [ DataContainer]]
    if isinstance(batch[0], Sequence):
        samples_per_gpu = len(batch[0]) * samples_per_gpu
        batch = sum(batch, [])
    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(stacked,
                                 batch[0].stack,
                                 batch[0].padding_value,
                                 cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(sample.data, pad,
                                  value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


def build_point_dataloader(dataset,
                           samples_per_gpu,
                           workers_per_gpu,
                           num_gpus=1,
                           dist=True,
                           shuffle=True,
                           seed=None,
                           **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = DistributedGroupSampler(dataset,
                                              samples_per_gpu,
                                              world_size,
                                              rank,
                                              seed=seed)
        else:
            sampler = DistributedSampler(dataset,
                                         world_size,
                                         rank,
                                         shuffle=False,
                                         seed=seed)
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=partial(
                                 multi_collate_fn,
                                 samples_per_gpu=samples_per_gpu),
                             pin_memory=False,
                             worker_init_fn=init_fn,
                             **kwargs)

    return data_loader
