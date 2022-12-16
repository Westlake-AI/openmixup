import torch
from PIL import Image
from .registry import DATASETS
from .base import BaseDataset
from .utils import to_numpy
from openmixup.utils import print_log


@DATASETS.register_module
class ClusterContrastiveDataset(BaseDataset):
    """Dataset for supervised or clustering + SSL augmentation (2N)
    """

    def __init__(self, data_source, pipeline, predefined_label=False, prefetch=False):
        super(ClusterContrastiveDataset, self).__init__(data_source, pipeline, prefetch)
        self.predefined_label = predefined_label
        # init clustering labels
        self.labels = [-1 for _ in range(self.data_source.get_length())]

    def __getitem__(self, idx):
        # pseudo labels
        if self.predefined_label == False:
            # clustering
            img = self.data_source.get_sample(idx)
            label = self.labels[idx]
        else:
            img, label = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        # contrastive learning data
        img1 = self.pipeline(img)
        img2 = self.pipeline(img)
        if self.prefetch:
            img1 = torch.from_numpy(to_numpy(img1))
            img2 = torch.from_numpy(to_numpy(img2))
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        # provide data, label + idx
        return dict(img=img_cat, pseudo_label=label, idx=idx)

    def assign_labels(self, labels):
        assert len(self.labels) == len(labels), \
            "Inconsistent lenght of asigned labels, \
            {} vs {}".format(len(self.labels), len(labels))
        self.labels = labels[:]

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
        assert self.predefined_label == True
        eval_res = {}

        target = torch.LongTensor(self.data_source.labels)
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
