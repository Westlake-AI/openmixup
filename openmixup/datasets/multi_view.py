import torch
from PIL import Image

from openmixup.utils import print_log, build_from_cfg
from torchvision.transforms import Compose

from .base import BaseDataset
from .registry import DATASETS, PIPELINES
from .builder import build_datasource
from .utils import to_numpy


@DATASETS.register_module()
class MultiViewDataset(BaseDataset):
    """The dataset outputs multiple views of an image.

    The number of views in the output dict depends on `num_views`. The
    image can be processed by one pipeline or multiple piepelines.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        num_views (list): The number of different views.
        pipelines (list[list[dict]]): A list of pipelines, where each pipeline
            contains elements that represents an operation defined in
            `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.

    Examples:
        >>> dataset = MultiViewDataset(data_source, [2], [pipeline])
        >>> output = dataset[idx]
        The output got 2 views processed by one pipeline.

        >>> dataset = MultiViewDataset(
        >>>     data_source, [2, 6], [pipeline1, pipeline2])
        >>> output = dataset[idx]
        The output got 8 views processed by two pipelines, the first two views
        were processed by pipeline1 and the remaining views by pipeline2.
    """

    def __init__(self, data_source, num_views, pipelines, prefetch=False):
        assert len(num_views) == len(pipelines)
        self.data_source = build_datasource(data_source)
        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipelines.append(pipeline)
        self.prefetch = prefetch

        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipelines[i]] * num_views[i])
        self.trans = trans

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        if self.data_source.return_label:
            img, target = img
        else:
            target = None
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        multi_views = list(map(lambda trans: trans(img), self.trans))
        if self.prefetch:
            multi_views = [
                torch.from_numpy(to_numpy(img)) for img in multi_views
            ]
        return dict(img=multi_views, gt_label=target, idx=idx)

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
        assert self.data_source.return_label and self.data_source.has_labels

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
