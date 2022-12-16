import os.path as osp
import mmcv
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info, load_checkpoint

from openmixup.datasets.registry import PIPELINES
from openmixup.models import build_model
from openmixup.models.utils import show_result
from openmixup.utils import (build_from_cfg, dist_forward_collect, 
                             nondist_forward_collect)


def init_model(config, checkpoint=None, device='cuda:0', options=None):
    """Initialize a model from config file for infererence.

    Args:
        config (str or `mmcv.Config`): Config file path or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        options (dict): Options to override some settings in the used config.

    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
    config.model.pretrained = None
    model = build_model(config.model)
    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_model(model, img):
    """Inference images with the model (classifier).

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if isinstance(img, str):
        img = Image.open(img)
        img = img.convert('RGB')
    else:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        else:
            if not isinstance(img, Image.Image):
                raise TypeError(f'Type {type(img)} cannot be recognized.')
    val_pipeline = Compose([
        build_from_cfg(p, PIPELINES) for p in cfg.data.val.pipeline])
    img = val_pipeline(img).unsqueeze(0)
    if next(model.parameters()).is_cuda:
        img = img.to(device)

    # forward the model
    with torch.no_grad():
        scores = model(img=img, mode="inference")
        pred_score = np.max(scores, axis=1)[0]
        pred_label = np.argmax(scores, axis=1)[0]
        result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
    return result


def multi_gpu_test(model, data_loader):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        dict(np.ndarray): The concatenated outputs with keys.
    """
    model.eval()
    func = lambda **x: model(mode='test', **x)
    rank, world_size = get_dist_info()
    results = dist_forward_collect(func, data_loader, rank,
                                   len(data_loader.dataset))
    return results


def single_gpu_test(model, data_loader):
    """Test model with local single gpu.

    This method tests model with a single gpu and supports showing results.

    Args:
        model (:obj:`torch.nn.Module`): Model to be tested.
        data_loader (:obj:`torch.utils.data.DataLoader`): Pytorch data loader.

    Returns:
        dict(np.ndarray): The concatenated outputs with keys.
    """
    model.eval()
    func = lambda **x: model(mode='test', **x)
    results = nondist_forward_collect(func, data_loader,
                                      len(data_loader.dataset))
    return results


def single_gpu_test_show(model,
                         data_loader,
                         show=False,
                         out_dir=None,
                         **show_kwargs):
    """Test model and show results with local single gpu.

    This method tests model with a single gpu and supports showing results.

    Args:
        model (:obj:`torch.nn.Module`): Model to be tested.
        data_loader (:obj:`torch.utils.data.DataLoader`): Pytorch data loader.
        show (bool): Whether to show the test results. Defaults to False.
        out_dir (str): The output directory of result plots of all samples.
            Defaults to None, which means not to write output files.
        **show_kwargs: Any other keyword arguments for showing results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    img_norm_cfg = dict(  # ImageNet as default
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(mode='inference', **data)

        batch_size = len(result)
        results.extend(result)

        if show or out_dir:
            scores = np.vstack(result)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            imgs = tensor2imgs(data['img'], **img_norm_cfg)

            for j, img in enumerate(imgs):
                if out_dir:
                    out_file = osp.join(out_dir, f"{i*batch_size + j}.png")
                else:
                    out_file = None

                result_show = {
                    'pred_score': pred_score[j],
                    'pred_label': pred_label[j],
                }
                show_result(
                    img,
                    result_show,
                    show=show,
                    out_file=out_file,
                    **show_kwargs)

        batch_size = data['img'].size(0)
        prog_bar.update(batch_size)

    return results
