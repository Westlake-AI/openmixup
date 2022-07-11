import platform
import random
import torch
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import digit_version
from torch.utils.data import DataLoader

from .sampler import DistributedSampler, DistributedGivenIterationSampler, RepeatAugSampler
from torch.utils.data import RandomSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     sampler='DistributedSampler',
                     shuffle=True,
                     replace=False,
                     seed=None,
                     pin_memory=True,
                     persistent_workers=True,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        imgs_per_gpu (int): (Deprecated, please use samples_per_gpu) Number of
            images on each GPU, i.e., batch size of each GPU. Defaults to None.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU. `persistent_workers` option needs num_workers > 0.
            Defaults to 1.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        replace (bool): Replace or not in random shuffle.
            It works on when shuffle is True. Defaults to False.
        seed (int): set seed for dataloader.
        pin_memory (bool, optional): If True, the data loader will copy Tensors
            into CUDA pinned memory before returning them. Defaults to True.
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0.
            Defaults to True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    if dist:
        rank, world_size = get_dist_info()
        if sampler == 'RepeatAugSampler':
            data_sampler = RepeatAugSampler(
                dataset, shuffle=shuffle, rank=rank)
        elif sampler == 'DistributedGivenIterationSampler':
            data_sampler = DistributedGivenIterationSampler(
                dataset, total_iter=kwargs.get('total_iter', 1e20),
                batch_size=kwargs.get('batch_size', 256), rank=rank)
        else:
            data_sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=shuffle, replace=replace)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        if replace:
            raise NotImplementedError
        data_sampler = RandomSampler(
            dataset) if shuffle else None  # TODO: set replace
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    # If sampler exists, turn off dataloader shuffle
    if data_sampler is not None:
        shuffle = False

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        kwargs['persistent_workers'] = persistent_workers

    if kwargs.get('prefetch') is not None:
        prefetch = kwargs.pop('prefetch')
        img_norm_cfg = kwargs.pop('img_norm_cfg')
    else:
        prefetch = False

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=data_sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        **kwargs)

    if prefetch:
        data_loader = PrefetchLoader(data_loader, img_norm_cfg['mean'], img_norm_cfg['std'])

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Function to initialize each worker.

    The seed of each worker equals to
    ``num_worker * rank + worker_id + user_seed``.

    Args:
        worker_id (int): Id for each worker.
        num_workers (int): Number of workers.
        rank (int): Rank in distributed training.
        seed (int): Random seed.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class PrefetchLoader:
    """A data loader wrapper for prefetching data."""

    def __init__(self, loader, mean, std):
        self.loader = loader
        self._mean = mean
        self._std = std

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True
        self.mean = torch.tensor([x * 255 for x in self._mean]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in self._std]).cuda().view(1, 3, 1, 1)

        for next_input_dict in self.loader:
            with torch.cuda.stream(stream):
                if isinstance(next_input_dict['img'], list):
                    next_input_dict['img'] = [
                        data.cuda(non_blocking=True).float().sub_(self.mean).div_(self.std)
                        for data in next_input_dict['img']
                    ]
                else:
                    data = next_input_dict['img'].cuda(non_blocking=True)
                    next_input_dict['img'] = data.float().sub_(self.mean).div_(self.std)

            if not first:
                yield input
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input_dict

        next_input_dict = None
        torch.cuda.empty_cache()
        yield input

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
