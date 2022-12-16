import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import time

from .utils import to_numpy
from openmixup.utils import print_log, build_from_cfg
from .registry import DATASETS, PIPELINES
from .builder import build_datasource


@DATASETS.register_module
class SemiSupervisedDataset(Dataset):
    """Dataset for semi-supervised methods
     *** using contrastive-based (SSL) augmentation (2N) ***
    
    Args:
        data_source_labeled (dict): Data source for labeled data.
        data_source_unlabeled (dict): Data source for unlabeled data.
        pipeline_labeled (list[dict]): A list of dict for the labeled (L) dataset, where
            each element represents an operation defined in `datasets.pipelines`.
        pipeline_unlabeled (list[dict]): A list of dict for the unlabeled (UL) dataset.
        pipeline_strong (list[dict]): A list of dict for additional stonge augmentations.
            If 'pipeline_strong' is not None, imply it to the second sample of L and UL.
        ret_samples (dict): Choice of return samples' shape, 'x_l_2' denotes the second
            labeled samples, 'x_ul_2' denotes the second unlabeled samples. For examples,
                'FixMatch' requires strong & weak augmented unlabeled pairs, as
                    ret_samples=dict(x_l_2=False, x_ul_2=True).
                'Self-Tuning' requires two ways of L and UL samples for the PGC loss, as
                    ret_samples=dict(x_l_2=True, x_ul_2=True).
            Default to None, i.e., return shape of [N, 4, C, H, W].
    """

    def __init__(self,
                data_source_labeled,
                data_source_unlabeled,
                pipeline_labeled=None,
                pipeline_unlabeled=None,
                pipeline_strong=None,
                ret_samples=dict(x_l_2=True, x_ul_2=True),
                prefetch=False):
        self.data_source_labeled = build_datasource(data_source_labeled)
        self.data_source_unlabeled = build_datasource(data_source_unlabeled)
        # labeled
        pipeline_labeled = [build_from_cfg(p, PIPELINES) for p in pipeline_labeled]
        self.pipeline_labeled = Compose(pipeline_labeled)
        # unlabeled
        pipeline_unlabeled = [build_from_cfg(p, PIPELINES) for p in pipeline_unlabeled]
        self.pipeline_unlabeled = Compose(pipeline_unlabeled)
        # strong aug
        self.pipeline_strong = None
        if pipeline_strong is not None:
            pipeline_strong = [build_from_cfg(p, PIPELINES) for p in pipeline_strong]
            self.pipeline_strong = Compose(pipeline_strong)

        self.pseudo_labels = [-1 for _ in range(self.data_source_unlabeled.get_length())]
        # length is dependent on the large dataset
        if self.data_source_labeled.get_length() >= self.data_source_unlabeled.get_length():
            self.length = self.data_source_labeled.get_length()
            # self.length_gap = self.length - self.data_source_unlabeled.get_length()
            self.unlabeled_large = False
        else:
            self.length = self.data_source_unlabeled.get_length()
            # self.length_gap = self.length - self.data_source_labeled.get_length()
            self.unlabeled_large = True
        self.ret_samples = ret_samples
        self.x_l_2 = ret_samples.get("x_l_2", True)
        self.x_ul_2 = ret_samples.get("x_ul_2", True)
        self.prefetch = prefetch

    def __len__(self):
        return self.length

    def assign_labels(self, labels):
        """ assign pseudo labels for the unlabeled data """
        assert len(self.pseudo_labels) == len(labels), \
            "Inconsistent lenght of asigned labels for unlabeled dataset, \
            {} vs {}".format(len(self.pseudo_labels), len(labels))
        self.pseudo_labels = labels[:]

    def __getitem__(self, idx):
        # Warning: the seed might be different in a mini-batch!!!
        seed = time.localtime()[3] * 10 + time.localtime()[4] % 10  # hour in [0,11], min in [0,59]
        # idx for labeled and unlabeled data
        if self.unlabeled_large == True:
            idx_labeled = (idx + seed) % self.data_source_labeled.get_length()
            idx_unlabeled = idx
        else:
            idx_labeled = idx
            idx_unlabeled = (idx + seed) % self.data_source_unlabeled.get_length()
        # labeled data: img + gt_labels
        img_labeled, gt_labels = self.data_source_labeled.get_sample(idx_labeled)
        # unlabeled data: img + pseudo labels
        img_unlabeled = self.data_source_unlabeled.get_sample(idx_unlabeled)
        pseudo_labels = self.pseudo_labels[idx_unlabeled]
        assert isinstance(img_labeled, Image.Image) and isinstance(img_unlabeled, Image.Image), \
            'The output from the data source must be an Image, got: {}, {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img_labeled), type(img_unlabeled))
        
        # contrastive-based pipelines
        img1_labeled = self.pipeline_labeled(img_labeled)
        if self.x_l_2 == True:
            if self.pipeline_strong is not None:
                img2_labeled = self.pipeline_strong(img_labeled)
            else:
                img2_labeled = self.pipeline_labeled(img_labeled)
        img1_unlabeled = self.pipeline_unlabeled(img_unlabeled)
        if self.x_ul_2 == True:
            if self.pipeline_strong is not None:
                img2_unlabeled = self.pipeline_strong(img_unlabeled)
            else:
                img2_unlabeled = self.pipeline_unlabeled(img_unlabeled)
        
        # prefetch as numpy
        if self.prefetch:
            img1_labeled = torch.from_numpy(to_numpy(img1_labeled))
            if self.x_l_2 == True:
                img2_labeled = torch.from_numpy(to_numpy(img2_labeled))
            img1_unlabeled = torch.from_numpy(to_numpy(img1_unlabeled))
            if self.x_ul_2 == True:
                img2_unlabeled = torch.from_numpy(to_numpy(img2_unlabeled))
        
        # returm samples
        if self.x_l_2:
            cat_list = [img1_labeled.unsqueeze(0), img2_labeled.unsqueeze(0)]
        else:
            cat_list = [img1_labeled.unsqueeze(0),]
        cat_list.append(img1_unlabeled.unsqueeze(0))
        if self.x_ul_2:
            cat_list.append(img2_unlabeled.unsqueeze(0))
        img = torch.cat(cat_list, dim=0)
        # provide data + labels
        return dict(img=img, gt_labels=gt_labels, ps_labels=pseudo_labels, gt_idx=idx_labeled)
    
    def evaluate(self, scores, keyword, logger=None, topk=(1, 5), **kwargs):
        """ Evaluation as supervised classification
        
        Args:
            scores (tensor): The prediction values of output heads in (N, \*).
            keyword (str): The corresponding head name in (N, \*).
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
        Returns:
            dict: evaluation results
        """
        eval_res = {}

        target = torch.LongTensor(self.data_source_labeled.labels)
        assert scores.size(0) == target.size(0), \
            "Inconsistent length for results and labels, {} vs {}".format(
            scores.size(0), target.size(0))
        num = scores.size(0)
        _, pred = scores.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # KxN
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0).item()
            acc = correct_k * 100.0 / num
            eval_res["{}_top{}".format(keyword, k)] = acc
            if logger is not None and logger != 'silent':
                print_log(
                    "{}_top{}: {:.03f}".format(keyword, k, acc),
                    logger=logger)
        return eval_res
