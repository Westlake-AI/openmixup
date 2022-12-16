import torch.nn as nn
from mmcv.runner import get_dist_info
from torch.utils.data import Dataset

from openmixup import datasets
from openmixup.models.utils import MultiPooling
from openmixup.utils import nondist_forward_collect, dist_forward_collect


class Extractor(object):
    """Feature extractor.

    Args:
        dataset (Dataset | dict): A PyTorch dataset or dict that indicates
            the dataset.
        imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
            each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        dist_mode (bool): Use distributed extraction or not. Default: False.
    """

    def __init__(self,
                 dataset,
                 imgs_per_gpu,
                 workers_per_gpu,
                 forward_mode='extract',
                 dist_mode=False,
                 **kwargs):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
        else:
            raise TypeError(
                f'dataset must be a Dataset object or a dict, not {type(dataset)}')
        self.data_loader = datasets.build_dataloader(
            self.dataset,
            imgs_per_gpu,
            workers_per_gpu,
            dist=dist_mode,
            shuffle=False,
            prefetch=kwargs.get('prefetch', False),
            img_norm_cfg=kwargs.get('img_norm_cfg', dict()),
        )
        assert forward_mode in ['test', 'vis', 'extract',]
        self.forward_mode = forward_mode
        self.dist_mode = dist_mode
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _forward_func(self, runner, **x):
        backbone_feat = runner.model(mode=self.forward_mode, **x)
        last_layer_feat = runner.model.module.neck([backbone_feat[-1]])[0]
        last_layer_feat = last_layer_feat.view(last_layer_feat.size(0), -1)
        return dict(feature=last_layer_feat.cpu())

    def __call__(self, runner):
        func = lambda **x: self._forward_func(runner, **x)
        if self.dist_mode:
            feats = dist_forward_collect(
                func,
                self.data_loader,
                runner.rank,
                len(self.dataset),
                ret_rank=-1)['feature']  # NxD
        else:
            feats = nondist_forward_collect(func, self.data_loader,
                                            len(self.dataset))['feature']
        return feats

    def extract(self, model, distributed=False):
        """The extract function to apply forward function and choose
        distributed or not."""
        model.eval()

        # the function sent to collect function
        def func(**x):
            return self._forward_func(model, **x)

        if distributed:
            rank, world_size = get_dist_info()
            results = dist_forward_collect(func, self.data_loader, rank,
                                           len(self.data_loader.dataset))
        else:
            results = nondist_forward_collect(func, self.data_loader,
                                              len(self.data_loader.dataset))
        return results


class MultiExtractProcess(object):
    """Multi-stage intermediate feature extraction process for `extract.py` and
    `tsne_visualization.py` in tools.

    This process extracts feature maps from different stages of backbone, and
    average pools each feature map to around 9000 dimensions.

    Args:
        pool_type (str): Pooling type in :class:`MultiPooling`. Options are
            "adaptive" and "specified". Defaults to "specified".
        backbone (str): Backbone type, now only support "resnet50".
            Defaults to "resnet50".
        layer_indices (Sequence[int]): Output from which stages.
            0 for stem, 1, 2, 3, 4 for res layers. Defaults to (0, 1, 2, 3, 4).
    """

    def __init__(self,
                 pool_type='specified',
                 backbone='resnet50',
                 layer_indices=(0, 1, 2, 3, 4)):
        self.multi_pooling = MultiPooling(
            pool_type, in_indices=layer_indices, backbone=backbone)
        self.layer_indices = layer_indices
        for i in self.layer_indices:
            assert i in [0, 1, 2, 3, 4]

    def _forward_func(self, model, **x):
        """The forward function of extract process."""
        backbone_feats = model(mode='extract', **x)
        pooling_feats = self.multi_pooling(backbone_feats)
        flat_feats = [xx.view(xx.size(0), -1) for xx in pooling_feats]
        feat_dict = {
            f'feat{self.layer_indices[i] + 1}': feat.cpu()
            for i, feat in enumerate(flat_feats)
        }
        return feat_dict

    def extract(self, model, data_loader, distributed=False):
        """The extract function to apply forward function and choose
        distributed or not."""
        model.eval()

        # the function sent to collect function
        def func(**x):
            return self._forward_func(model, **x)

        if distributed:
            rank, world_size = get_dist_info()
            results = dist_forward_collect(func, data_loader, rank,
                                           len(data_loader.dataset))
        else:
            results = nondist_forward_collect(func, data_loader,
                                              len(data_loader.dataset))
        return results
