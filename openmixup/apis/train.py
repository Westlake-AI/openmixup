import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_runner, DistSamplerSeedHook, get_dist_info

from openmixup.datasets import build_dataloader
from openmixup.core.optimizers import build_optimizer
from openmixup.core.hooks import (build_hook, build_addtional_scheduler,
                                  DistOptimizerHook, Fp16OptimizerHook)
from openmixup.utils import find_latest_checkpoint, get_root_logger, print_log

# import fp16 supports
try:
    import apex
    default_fp16 = 'apex'
except ImportError:
    default_fp16 = 'mmcv'
    warnings.warn('DeprecationWarning: Nvidia Apex is not installed, '
                  'using FP16OptimizerHook modified from mmcv.')


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            sampler=getattr(cfg, 'sampler', 'DistributedSampler'),
            shuffle=True,
            replace=getattr(cfg.data, 'sampling_replace', False),
            seed=cfg.seed,
            drop_last=getattr(cfg.data, 'drop_last', False),
            prefetch=getattr(cfg, 'prefetch', False),
            persistent_workers=getattr(cfg, 'persistent_workers', True),
            img_norm_cfg=cfg.img_norm_cfg) for ds in dataset
    ]

    # if you have addtional_scheduler, select chosen params
    if cfg.get('addtional_scheduler', None) is not None:
        param_names = dict(model.named_parameters()).keys()
        assert isinstance(cfg.optimizer.get('paramwise_options', False), dict)

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # fp16 and optimizer
    if distributed:
        if cfg.get('use_fp16', False):
            # fp16 settings
            fp16_cfg = cfg.get('fp16', dict(type='apex'))
            fp16_cfg['type'] = fp16_cfg.get('type', default_fp16)
            if fp16_cfg['type'] == 'apex':
                model, optimizer = apex.amp.initialize(
                    model.cuda(), optimizer, opt_level="O1")
                optimizer_config = DistOptimizerHook(
                    **cfg.optimizer_config, use_fp16=True)
                print_log('**** Initializing mixed precision apex done. ****')
            elif fp16_cfg['type'] == 'mmcv':
                loss_scale = fp16_cfg.get('loss_scale', 'dynamic')
                optimizer_config = Fp16OptimizerHook(
                    **cfg.optimizer_config, loss_scale=loss_scale, distributed=True)
                print_log('**** Initializing mixed precision mmcv done. ****')
        else:
            optimizer_config = DistOptimizerHook(**cfg.optimizer_config, use_fp16=False)
    else:
        optimizer_config = cfg.optimizer_config

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model if next(model.parameters()).is_cuda else model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids).cuda()

    if cfg.get('runner') is None:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    # build runner
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register custom hooks: bofore ValidationHook and CheckpointSaverHook
    for hook in cfg.get('custom_hooks', list()):
        common_params = dict(dist_mode=distributed)
        if hook.type == "DeepClusterAutoMixHook" or hook.type == "DeepClusterHook":
            common_params = dict(dist_mode=distributed, data_loaders=data_loaders)
        hook_cfg = hook.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        runner.register_hook(build_hook(hook, common_params), priority=priority)

    # register basic hooks
    runner.register_training_hooks(cfg.lr_config,
                                   optimizer_config,
                                   cfg.checkpoint_config,
                                   cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register custom optional_scheduler hook
    if cfg.get('addtional_scheduler', None) is not None:
        runner.register_hook(
            build_addtional_scheduler(param_names, cfg.addtional_scheduler))

    # register evaluation hook
    if cfg.get('evaluation', None):
        eval_cfg = cfg.get('evaluation', dict())
        eval_cfg = dict(
            type='ValidateHook',
            dataset=cfg.data.val,
            dist_mode=distributed,
            initial=eval_cfg.get('initial', True),
            interval=eval_cfg.get('interval', 1),
            save_val=eval_cfg.get('save_val', False),
            imgs_per_gpu=eval_cfg.get('imgs_per_gpu', cfg.data.imgs_per_gpu),
            workers_per_gpu=eval_cfg.get('imgs_per_gpu', cfg.data.workers_per_gpu),
            eval_param=eval_cfg.get('eval_param', dict(topk=(1, 5))),
            prefetch=cfg.data.val.get('prefetch', False),
            img_norm_cfg=cfg.img_norm_cfg,
        )
        # We use `ValidationHook` instead of `EvalHook` in mmcv. `EvalHook` needs to be
        # executed after `IterTimerHook`, or it will cause a bug if use `IterBasedRunner`.
        runner.register_hook(build_hook(eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    cfg.workflow = [tuple(x) for x in cfg.workflow]
    runner.run(data_loaders, cfg.workflow)
