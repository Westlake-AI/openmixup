import argparse
import math
import os
import mmcv
import numpy as np

import importlib
import os
import os.path as osp

import torch

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from einops import rearrange

from mmcv import Config, DictAction
from mmcv.utils import to_2tuple
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import (MMDataParallel, MMDistributedDataParallel,
                           is_module_wrapper)

from torch.nn import BatchNorm1d, BatchNorm2d, GroupNorm, LayerNorm
from torchvision.transforms import Compose

from openmixup import digit_version
from openmixup.datasets import build_dataset, build_dataloader, PIPELINES
from openmixup.models import build_model
from openmixup.utils import build_from_cfg

from openmixup.utils import (dist_forward_collect, setup_multi_processes,
                             nondist_forward_collect, traverse_replace)

try:
    from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM,
                                  GradCAMPlusPlus, LayerCAM, XGradCAM)
    from pytorch_grad_cam.activations_and_gradients import \
        ActivationsAndGradients
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
                      '3rd party package pytorch_grad_cam.')

# set of transforms, which just change data format, not change the pictures
FORMAT_TRANSFORMS_SET = {'ToTensor', 'Normalize',}

# Supported feature map visualization methods
METHOD_MAP = {
    'saliency': None,
    'feature': None,
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
}


def flatten(xs_list):
    return [x for xs in xs_list for x in xs]


def fourier(x):  # 2D Fourier transform
    f = torch.fft.fft2(x)
    f = f.abs() + 1e-6
    f = f.log()
    return f


def fft_shift(x):  # shift Fourier transformed feature map
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2, 3))


def make_segments(x, y):  # make segment for `plot_segment`
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_segment(ax, xs, ys, marker, liner='solid', cmap_name="plasma", alpha=1.0, zorder=1):
    # plot with cmap segments
    z = np.linspace(0.0, 1.0, len(ys))
    z = np.asarray(z)
    
    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(0.0, 1.0)
    segments = make_segments(xs, ys)
    lc = LineCollection(segments, array=z, cmap=cmap_name, norm=norm,
                        linewidth=2.0, linestyles=liner, alpha=alpha)
    ax.add_collection(lc)

    colors = [cmap(x) for x in xs]
    ax.scatter(xs, ys, color=colors, marker=marker, zorder=100 + zorder)
    return lc


def create_cmap(color_name, end=0.95):
    """ create custom cmap """
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    color = cm.get_cmap(color_name, 200)
    if end == 0.8:
        newcolors = color(np.linspace(0.75, end, 200))
    else:
        newcolors = color(np.linspace(max(0.5, end-0.4), end, 200))
    newcmp = ListedColormap(newcolors, name=color_name+"05_09")
    return newcmp


def make_proxy(color, marker, liner, **kwargs):
    """ add custom legend """
    from matplotlib.lines import Line2D
    cmap = cm.get_cmap(color)
    color = cmap(np.arange(4) / 4)
    return Line2D([0, 1], [0, 1], color=color[3], marker=marker, linestyle=liner)


def plot_fourier_features(latents):
    # Fourier transform feature maps
    fourier_latents = []
    for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
        latent = latent.cpu()
        
        if len(latent.shape) == 3:  # for ViT
            b, n, c = latent.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
        elif len(latent.shape) == 4:  # for CNN
            b, c, h, w = latent.shape
        else:
            raise Exception("shape: %s" % str(latent.shape))
        latent = fourier(latent)
        latent = fft_shift(latent).mean(dim=(0, 1))
        latent = latent.diag()[int(h/2):]  # only use the half-diagonal components
        latent = latent - latent[0]  # visualize 'relative' log amplitudes 
                                    # (i.e., low-freq amp - high freq amp)
        fourier_latents.append(latent)
    
    return fourier_latents


def plot_channel_features(latents):
    # Fourier transform feature maps
    fourier_latents = []
    for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
        latent = latent.cpu()
        
        if len(latent.shape) == 3:  # for ViT
            b, n, c = latent.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
        elif len(latent.shape) == 4:  # for CNN
            b, c, h, w = latent.shape
        else:
            raise Exception("shape: %s" % str(latent.shape))
        latent = fourier(latent)
        latent = fft_shift(latent).mean(dim=(0, 1))
        latent = latent.diag()[int(h/2):]  # only use the half-diagonal components
        latent = latent - latent[0]  # visualize 'relative' log amplitudes 
                                    # (i.e., low-freq amp - high freq amp)
        fourier_latents.append(latent)
    
    return fourier_latents


def plot_variance_features(latents):
    # aggregate feature map variances
    variances = []
    for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
        latent = latent.cpu()
        
        if len(latent.shape) == 3:  # for ViT
            b, n, c = latent.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
        elif len(latent.shape) == 4:  # for CNN
            b, c, h, w = latent.shape
        else:
            raise Exception("shape: %s" % str(latent.shape))
        variances.append(latent.var(dim=[-1, -2]).mean(dim=[0, 1]))
    
    return variances


def plot_fft_A(fourier_latents, save_path, save_format='png', drop_last=False):
    # A. Plot Fig 2a: "Relative log amplitudes of Fourier transformed feature maps"
    fig, ax1 = plt.subplots(1, 1, figsize=(3.3, 4), dpi=150)
    num_layer = len(fourier_latents)
    if drop_last:
        fourier_latents = fourier_latents[:-1]
    for i, latent in enumerate(reversed(fourier_latents)):
        freq = np.linspace(0, 1, len(latent))
        ax1.plot(freq, latent, color=cm.plasma_r(i / num_layer))

    ax1.set_xlim(left=0, right=1)
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("$\Delta$ Log amplitude")

    from matplotlib.ticker import FormatStrFormatter
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1fπ'))

    plt.show()
    plt.savefig(os.path.join(save_path, f'fft_features.{save_format}'))
    plt.close()


def set_plot_args(model_name, idx=0, alpha_base=0.9):
    # setup
    linear_mapping = dict(cl="dashed", mim="solid", ssl="dashed", sl="dashdot")
    marker_mapping = dict(cl="s", mim="p", ssl="D", sl="o")
    colour_mapping = dict(cl=["YlGnBu", "Blues", "GnBu", "Greens", "YlGn", "winter"],
                            # mim=["Reds", "OrRd", "YlOrRd", "RdPu",],  # ResNet
                            mim=["Reds", "YlOrRd", "OrRd", "RdPu",],  # ViT
                            ssl=["PuRd",],  # red
                            sl=["autumn", "winter", ],
                        )
    zorder_mapping = dict(cl=3, mim=4, ssl=2, sl=1)

    prefix = model_name.split("_")[0]
    alpha = alpha_base if prefix != 'sl' else 0.7
    marker = marker_mapping[prefix]
    liner = linear_mapping[prefix]
    cmap_list = colour_mapping[prefix]
    cmap_name = create_cmap(cmap_list[idx % len(cmap_list)], end=0.8 if prefix == 'sl' else 0.95)
    zorder = zorder_mapping[prefix]
    # refine model_name
    model_name = model_name.split("_")[-1].replace("+", " \ ")
    model_name = r"$\mathrm{" + model_name + "}$"
    
    return model_name, alpha, marker, liner, cmap_name, zorder


def plot_fft_B(args, fourier_latents, save_path, model_names=None, save_format='png'):
    # B. Plot Fig 8: "Relative log amplitudes of high-frequency feature maps"

    # plot settings
    alpha_base = 0.9
    font_size = 13
    cmap_name = "plasma"
    liner = "solid"

    if model_names is None:
        dpi = 120
        model_names = ['ssl_' + args.model_name]
        fourier_latents = [fourier_latents]
    else:
        dpi = 400
        assert isinstance(model_names, list) and len(model_names) >= 1
        zipped = zip(model_names, fourier_latents)
        zipped = sorted(zipped, key=lambda x:x[0])
        zipped = zip(*zipped)
        model_names, fourier_latents = [list(x) for x in zipped]
    
    fig, ax2 = plt.subplots(1, 1, figsize=(6.5, 5), dpi=dpi)
    proxy_list = []
    for i in range(len(model_names)):
        print(i, model_names[i], len(fourier_latents[i]))
        if "resnet" in args.model_name:
            pools = [4, 8, 14]
            msas = []
            marker = "D"
        elif "vit" in args.model_name or "deit" in args.model_name:
            pools = []
            msas = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,]  # vit-tiny
            marker = "o"
        else:
            import warnings
            warnings.warn("The configuration for %s are not implemented." % args.model_name, Warning)
            pools, msas = [], []
            marker = "s"
        
        # setup
        model_names[i], alpha, marker, liner, cmap_name, zorder = set_plot_args(model_names[i], i, alpha_base)
        # add legend
        proxy_list.append(make_proxy(cmap_name, marker, liner, linewidth=2))

        # Normalize
        depths = range(len(fourier_latents[i]))
        depth = len(depths) - 1
        depths = (np.array(depths)) / depth
        pools = (np.array(pools)) / depth
        msas = (np.array(msas)) / depth

        lc = plot_segment(ax2, depths, [latent[-1] for latent in fourier_latents[i]],
                     marker=marker, liner=liner, alpha=alpha, cmap_name=cmap_name, zorder=zorder)

    # ploting
    for pool in pools:
        ax2.axvspan(pool - 1.0 / depth, pool + 0.0 / depth, color="tab:blue", alpha=0.15, lw=0)
    for msa in msas:
        ax2.axvspan(msa - 1.0 / depth, msa + 0.0 / depth, color="tab:gray", alpha=0.15, lw=0)

    ax2.set_xlabel(r"$\mathrm{Normalized \ Depth}$", fontsize=font_size+2)
    ax2.set_ylabel(r"$\mathrm{\Delta \ Log \ Amplitude}$", fontsize=font_size+2)
    ax2.set_xlim(-0.01, 1.01)

    if len(model_names) > 1:
        # ax2.legend(proxy_list, model_names, loc='upper left', fontsize=font_size)
        ax2.legend(proxy_list, model_names, fontsize=font_size)
        plt.grid(ls='--', alpha=0.5, axis='y')

    from matplotlib.ticker import FormatStrFormatter
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.show()
    plt.savefig(
        os.path.join(save_path, f'high_freq_fft_features.{save_format}'),
        dpi=dpi, bbox_inches='tight', format=save_format)
    plt.close()


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

    parser.add_argument('--vis_layers', type=int, default=4,
                        help='The number of layers to visualize')

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
    out_indice_list = [i for i in range(args.vis_layers)]
    cfg.model.backbone.out_indices = out_indice_list

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

    save_path = osp.join(osp.abspath(cfg.work_dir), 'latents.pt')
    if osp.exists(save_path):
        latents = torch.load(save_path)
    # forward latents
    else:
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

        # load a sample ImageNet-1K image -- use the full val dataset for precise results
        latents = dict()
        with torch.no_grad():
            for i, data in tqdm(enumerate(data_loader)):
                if isinstance(data, tuple):
                    assert len(data) == 2
                    img, label = data
                else:
                    assert isinstance(data, dict)
                    img = data['img']
                img = img.cuda()

                if is_module_wrapper(model):
                    outs = model.module.backbone(img)
                else:
                    outs = model.backbone(img)
                # accumulate `latents` by collecting hidden states of a model
                for b in range(args.vis_layers):
                    feat = outs[b]
                    # print(b, feat.shape)
                    if feat.dim() == 4:
                        if not feat.shape[2] == feat.shape[3]: # (B, H, W, C) to (B, C, H, W)
                            feat = feat.permute(0, 3, 1, 2)
                    else:
                        B, L, C = feat.shape
                        H = int(math.sqrt(L))
                        feat = feat.permute(0, 2, 1).reshape(B, C, H, H)
                    if i == 0:
                        latents[str(b)] = list()
                        latents[str(b)].append(feat.detach().cpu())
                    else:
                        latents[str(b)].append(feat.detach().cpu())
                if i == 10:
                    break

        latent_list = list()
        for i in range(args.vis_layers):
            l = torch.cat(latents[str(i)], dim=0)
            latent_list.append(l)
        latents = latent_list
        torch.save(latents, save_path)

    fft_latents = plot_fourier_features(latents)
    plot_fft_A(fft_latents, osp.abspath(cfg.work_dir), args.plot_format)


if __name__ == '__main__':
    main()
