import torch
import numpy as np
from PIL import Image

from openmixup.models.utils import precision_recall_f1, support
from openmixup.utils import build_from_cfg, print_log

from .registry import DATASETS, PIPELINES
from .base import BaseDataset
from torchvision.transforms import Compose
from .utils import to_numpy
try:
    from skimage.feature import hog, local_binary_pattern
except:
    print("Please install scikit-image.")


@DATASETS.register_module
class MaskedImageDataset(BaseDataset):
    """The dataset outputs a processed image with mask for Masked Image Modeling.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of basic augmentations dict, where each element
            represents an operation defined in `mmselfsup.datasets.pipelines`.
        mask_pipeline (list[dict]): A list of mask generation dict.
        feature_mode (str): Mode of predefined feature extraction as MIM targets.
        feature_args (dict): A args dict of feature extraction. Defaults to None.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self,
                 data_source,
                 pipeline, mask_pipeline=None,
                 feature_mode=None, feature_args=dict(),
                 prefetch=False):
        super(MaskedImageDataset, self).__init__(data_source, pipeline, prefetch)
        self.mask_pipeline = mask_pipeline
        self.feature_mode = feature_mode
        assert self.feature_mode in [None, 'hog', 'lbp',]
        if self.mask_pipeline is not None:
            if self.feature_mode in ['hog', 'lbp']:
                assert prefetch == True, "Feature extraction needs uint8 images."
            else:
                assert prefetch == False, "Turn off `prefetch` when use RGB target."
            mask_pipeline = [build_from_cfg(p, PIPELINES) for p in mask_pipeline]
            self.mask_pipeline = Compose(mask_pipeline)
        self.return_label = self.data_source.return_label
        
        if self.feature_mode == 'hog':
            self.feature_args = dict(
                orientations=feature_args.get('orientations', 9),
                pixels_per_cell=feature_args.get('pixels_per_cell', (8, 8)),
                cells_per_block=feature_args.get('cells_per_block', (1, 1)),
                feature_vector=False, visualize=False, multichannel=True,
            )
        elif self.feature_mode == 'lbp':
            self.feature_args = dict(
                P=feature_args.get('P', 8),
                R=feature_args.get('R', 8),
                method=feature_args.get('method', 'nri_uniform'),
            )

    def __getitem__(self, idx):
        ret_dict = dict(idx=idx)
        if self.return_label:
            img, target = self.data_source.get_sample(idx)
            ret_dict['gt_label'] = target
        else:
            img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))

        # process img
        img = self.pipeline(img)
        img_mim = None
        
        if self.mask_pipeline is not None:
            # get predefined masks
            mask = self.mask_pipeline(img)
            if isinstance(mask, tuple):
                img_mim, mask = mask
            # get features with processed img (not ToTensor)
            if self.feature_mode is not None:
                if self.feature_mode == 'hog':
                    feat = hog(np.array(img, dtype=np.uint8), **self.feature_args).squeeze()
                elif self.feature_mode == 'lbp':
                    feat = local_binary_pattern(
                        np.array(img.convert('L'), dtype=np.uint8).squeeze(), **self.feature_args)
                    feat = np.expand_dims(feat, axis=2) / np.max(feat)
                # [H, W, 3] -> [C', H', W']
                feat = torch.from_numpy(feat).type(torch.float32).permute(2, 0, 1)
                mask = [mask, feat]  # extracted feature as the MIM target
            else:
                mask = [mask, img]  # raw RGB img as the MIM target
            ret_dict['mask'] = mask

        # update masked img as the input
        if img_mim is not None:
            img = img_mim
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        ret_dict['img'] = img

        return ret_dict

    def evaluate(self,
                 scores, keyword, logger=None,
                 metric='accuracy', metric_options=None, topk=(1, 5),
                 **kwargs):
        """The evaluation function to output accuracy (supervised).

        Args:
            scores (tensor): The prediction values of output heads in (N, \*).
            keyword (str): The corresponding head name in (N, \*).
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            metric (str | list[str]): Metrics to be evaluated. Default to `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'thrs' and 'average_mode'. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = dict(average_mode='macro')
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        eval_res = {}
        eval_log = []
        allowed_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'support',]
        average_mode = metric_options.get('average_mode', 'macro')
        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')
        
        target = torch.LongTensor(self.data_source.labels)
        assert scores.size(0) == target.size(0), \
            "Inconsistent length for results and labels, {} vs {}".format(
            scores.size(0), target.size(0))
        
        if 'accuracy' in metrics:
            _, pred = scores.topk(max(topk), dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))  # KxN
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0).item()
                acc = correct_k * 100.0 / scores.size(0)
                eval_res[f"{keyword}_top{k}"] = acc
                eval_log.append("{}_top{}: {:.03f}".format(keyword, k, acc))
        
        if 'support' in metrics:
            support_value = support(scores, target, average_mode=average_mode)
            eval_res[f'{keyword}_support'] = support_value
            eval_log.append("{}_support: {:.03f}".format(keyword, support_value))
        
        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            thrs = metric_options.get('thrs', 0.)
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    scores, target, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    scores, target, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_res.update({f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_res[key] = values
                        eval_log.append("{}_{}: {:.03f}".format(keyword, key, values))
        
        if logger is not None and logger != 'silent':
            for _log in eval_log:
                print_log(_log, logger=logger)
        
        return eval_res
