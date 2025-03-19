import argparse
import importlib
import os
import os.path as osp
import time
import numpy as np

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from openmixup.datasets import build_dataloader, build_dataset
from openmixup.models import build_model
from openmixup.utils import (get_root_logger, dist_forward_collect, print_log,
                             setup_multi_processes, nondist_forward_collect, traverse_replace, occlusion_forward_collect,)


def single_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    results = nondist_forward_collect(func, data_loader,
                                      len(data_loader.dataset))
    return results

def single_gpu_occlusion_test(model, data_loader, drop_ratio, drop_size):

    assert 224 % drop_size == 0, f"drop size {drop_size} not compatible with 224 image"
    model.eval()
    func = lambda **x: model(mode='test', **x)
    results = occlusion_forward_collect(func, data_loader,
                                      len(data_loader.dataset), drop_ratio, drop_size)
    return results

def multi_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    rank, world_size = get_dist_info()
    results = dist_forward_collect(func, data_loader, rank,
                                   len(data_loader.dataset))
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', type=str,
                        default=None,
                        help='test config file path')
    parser.add_argument('--checkpoint', type=str,
                        default=None,
                        help='checkpoint file')
    parser.add_argument('--drop_size', type=int, default=16, help='drop images size')
    parser.add_argument('--max_ratio', type=float, default=None, help='drop image extend, from 0.0 to 1.0')
    parser.add_argument(
        '--work_dir',
        type=str,
        default='work_dirs/occlusion',
        help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
        help='port only works when launcher=="slurm"')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])
    cfg.gpu_ids = [args.gpu_id]

    cfg.model.pretrained = None  # ensure to use checkpoint rather than pretraining

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'test_{}_occlusion.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # creat a ratio list
    if args.max_ratio > 1.0:
        args.max_ratio = 1.0
    ratios = np.arange(0, args.max_ratio + 1e-4, 0.1)
    ratios.tolist()

    if not distributed:
        print_log("Occlusion Experiment Results.\n", logger=logger)
        model = MMDataParallel(model, device_ids=[0])
        for i in range(len(ratios)):
            print_log("Occlusion Ratio:{:.1f}%".format(ratios[i] * 100), logger=logger)
            outputs = single_gpu_occlusion_test(model, data_loader, ratios[i], args.drop_size)

            rank, _ = get_dist_info()
            if rank == 0:
                for name, val in outputs.items():
                    dataset.evaluate(
                        torch.from_numpy(val), name, logger, topk=(1, 5))



if __name__ == '__main__':
    main()