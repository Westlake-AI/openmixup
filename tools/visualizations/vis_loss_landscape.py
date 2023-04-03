"""
Visualizing the Loss Landscape of Neural Nets (NeurIPS, 2018).
    Modified from https://github.com/tomgoldstein/loss-landscape

Example command (plotting loss 1D surface):
bash tools/visualizations/dist_vis_loss.sh [PATH/to/config] 1 [PATH/to/checkpoint] \
    --x=-1:1:51 --dir_type weights --xnorm filter --xignore biasbn
"""

import argparse
import importlib
import os
import os.path as osp
import copy
import numpy as np
try:
    import h5py
except ImportError:
    h5py = None  # pip install h5py

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import (MMDataParallel, MMDistributedDataParallel,
                           is_module_wrapper)
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

import openmixup.utils.loss_landscape_utils as helper
from openmixup.datasets import build_dataloader, build_dataset
from openmixup.models import build_model
from openmixup.models.utils import accuracy
from openmixup.utils import (dist_forward_collect, setup_multi_processes,
                             nondist_forward_collect, traverse_replace)


""" https://github.com/tomgoldstein/loss-landscape/plot_surface.py
    Calculate and visualize the loss surface.
"""

def name_surface_file(args, dir_file):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # avoid `BlockingIOError` in h5f.open
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    return surf_file + ".h5"


def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file


def crunch_surface(surf_file, net, net_w, net_s, net_d,
                   data_loader, dataset, loss_key, acc_key, args=None):
    """ Calculate the loss values and accuracies of modified models in DDP. """

    rank, _ = get_dist_info()
    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates), len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = helper.get_job_indices(losses, xcoordinates, ycoordinates, comm=None)
    print('Computing %d values for rank %d'% (len(inds), rank))

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            helper.set_weights(net, net_w, net_d, coord)
        elif args.dir_type == 'states':
            helper.set_states(net, net_s, net_d, coord)

        if not args.distributed:
            outputs = single_gpu_test(net, data_loader)
        else:
            outputs = multi_gpu_test(net, data_loader)  # dict{key: np.ndarray}

        if rank == 0:
            for name, val in outputs.items():
                scores  = torch.from_numpy(val)
                targets = torch.LongTensor(dataset.data_source.labels)
                if is_module_wrapper(net):
                    loss = net.module.head.criterion(scores, targets)
                else:
                    loss = net.head.criterion(scores, targets)
                acc  = accuracy(scores, targets, topk=1)
                break

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()
            print('\nEvaluating  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f' % (
                    count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                    acc_key, acc))

    f.close()


def single_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    results = nondist_forward_collect(func, data_loader,
                                      len(data_loader.dataset))
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
        description='Plotting loss landscape by testing (and eval) models')
    # Testing setups
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', type=str, default='',
                        help='checkpoint file as the "model_file1"')
    parser.add_argument('--work_dir',
                        type=str,
                        default=None,
                        help='the dir to save logs and models')
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0,
                        help='id of gpu to use (only applicable to non-distributed testing)')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
                        help='port only works when launcher=="slurm"')
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file. If the value to '
                        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                        'Note that the quotation marks are necessary and that no white space '
                        'is allowed.')

    # Loss surface setups
    parser.add_argument('--plot_mode',
                        choices=['surface', 'trajectory', 'surface+trajectory'],
                        type=str, default='surface',
                        help='plot mode of loss landscape (defaults to "surface")')
    parser.add_argument('--model_file2', type=str, default='',
                        help='using (model_file2 - model_file1) as the xdirection')
    parser.add_argument('--model_file3', type=str, default='',
                        help='using (model_file3 - model_file1) as the ydirection')
    parser.add_argument('--dir_file', type=str, default='',
                        help='specify the name of direction file, or the path to a direction file')
    parser.add_argument('--dir_type', type=str, default='weights',
                        help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', type=str, default='-1:1:51',
                        help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', type=str, default=None,
                        help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', type=str, default='',
                        help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', type=str, default='',
                        help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', type=str, default='',
                        help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', type=str, default='',
                        help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False,
                        help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int,
                        help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', type=str, default='',
                        help='customize the name of surface file, could be an existing file.')
    parser.add_argument('--proj_file', type=str, default='',
                        help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float,
                        help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float,
                        help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float,
                        help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float,
                        help='plot contours every vlevel')
    parser.add_argument('--log', action='store_true', default=False,
                        help='Whether to use log scale for loss values')
    parser.add_argument('--plot_format', type=str, default='png',
                        help='The save format of plotted matplotlib images')

    # Loss trajectory setups
    parser.add_argument('--model_folder', default=None, type=str,
                        help='folders for models to be projected (defaults to work_dirs)')
    parser.add_argument('--prefix', default='epoch', type=str,
                        help='prefix for the checkpint model for plotting the trajectory')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='min index of epochs for plotting the trajectory')
    parser.add_argument('--max_epoch', default=200, type=int,
                        help='max number of epochs for plotting the trajectory')
    parser.add_argument('--save_interval', default=1, type=int,
                        help='interval to save models for plotting the trajectory')

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
    args.work_dir = cfg.work_dir
    cfg.gpu_ids = [args.gpu_id]

    cfg.model.pretrained = None  # ensure to use checkpoint rather than pretraining

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=args.distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not args.distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    rank, _ = get_dist_info()

    #--------------------------------------------------------------------------
    # Calculate and visualize the loss surface (1D or 2D) in `plot_surface.py`
    #--------------------------------------------------------------------------
    if 'surface' in args.plot_mode:
        try:
            args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]  # or it will cause TypeError
            args.xnum = int(args.xnum)
            args.ymin, args.ymax, args.ynum = (None, None, None)
            if args.y:
                args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
                args.ynum = int(args.ynum)
                assert args.ymin and args.ymax and args.ynum, \
                    'You specified some arguments for the y axis, but not all'
        except:
            raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

        model_w = helper.get_weights(model) # initial parameters
        model_s = copy.deepcopy(model.state_dict()) # deepcopy since state_dict are references
        dir_file = helper.name_direction_file(args) # name the direction file
        helper.setup_direction(args, dir_file, model)
        surf_file = name_surface_file(args, dir_file)
        setup_surface_file(args, surf_file, dir_file)

        # load directions
        model_d = helper.load_directions(dir_file)
        # calculate the consine similarity of the two directions
        if len(model_d) == 2:
            similarity = helper.cal_angle(
                helper.nplist_to_tensor(model_d[0]), helper.nplist_to_tensor(model_d[1]))
            print('cosine similarity between x-axis and y-axis: %f' % similarity)

        crunch_surface(surf_file, model, model_w, model_s, model_d, data_loader=data_loader, dataset=dataset,
            loss_key='train_loss', acc_key='train_acc', args=args)

        if rank == 0:
            if args.y:
                helper.plot_2d_contour(
                    surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, format=args.plot_format)
            else:
                helper.plot_1d_loss_err(
                    surf_file, args.xmin, args.xmax, args.loss_max, args.log, format=args.plot_format)

    #--------------------------------------------------------------------------
    # Caculate a projected optimization trajectory in `plot_trajectory.py` (2D)
    #--------------------------------------------------------------------------
    if 'trajectory' in args.plot_mode:
        # the loaded ckpt as the final model
        model_w = helper.get_weights(model) # initial parameters
        model_s = copy.deepcopy(model.state_dict()) # deepcopy since state_dict are references
        dir_file = helper.name_direction_file(args) # name the direction file

        # collect models to be projected
        if args.model_folder is None:
            args.model_folder = cfg.work_dir
        model_files = []
        for epoch in range(args.start_epoch, args.max_epoch + args.save_interval, args.save_interval):
            model_file = args.model_folder + f'/{args.prefix}_{str(epoch)}.pth'
            assert os.path.isfile(model_file), 'model %s does not exist' % model_file
            model_files.append(model_file)

        # load or create projection directions
        if args.dir_file:
            dir_file = args.dir_file
        else:
            dir_file = helper.setup_PCA_directions(args, model, model_files, model_w, model_s)

        # projection trajectory to given directions
        proj_file = helper.project_trajectory(dir_file, model_w, model_s, model, model_files,
                                              dir_type=args.dir_type, proj_method='cos')
        if rank == 0:
            helper.plot_trajectory(proj_file, dir_file, format=args.plot_format)

    if args.plot_mode == 'surface+trajectory':
        args.proj_file = proj_file if not args.proj_file else args.proj_file
        if rank == 0:
            assert args.y
            helper.plot_contour_trajectory(
                surf_file, dir_file, args.proj_file, 'train_loss', format=args.plot_format)


if __name__ == '__main__':
    main()
