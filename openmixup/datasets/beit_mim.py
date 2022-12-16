import torch
from PIL import Image

from openmixup.utils import build_from_cfg
from torchvision.transforms import Compose

from .registry import DATASETS, PIPELINES
from .base import BaseDataset
from .builder import build_datasource
from .utils import to_numpy


@DATASETS.register_module
class BEiTDataset(BaseDataset):
    """The dataset outputs two views of an image for BEiT.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipelines (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        mask_pipeline (list[dict]): A list of mask generation dict.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self,
                 data_source,
                 pipelines,
                 mask_pipeline,
                 prefetch=False):
        data_source['return_label'] = False
        self.data_source = build_datasource(data_source)
        self.prefetch = prefetch

        assert len(pipelines) == 2
        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipelines.append(pipeline)
        assert prefetch == False, "Turn off `prefetch` when use RGB target."
        mask_pipeline = Compose([
            build_from_cfg(p, PIPELINES) for p in mask_pipeline])
        trans = [self.pipelines[0], self.pipelines[1], mask_pipeline]
        self.trans = trans

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        assert isinstance(img, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(img))
        multi_views = list(map(lambda trans: trans(img), self.trans))
        if self.prefetch:
            multi_views = [
                torch.from_numpy(to_numpy(img)) for img in multi_views
            ]
        return dict(img=multi_views, idx=idx)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplementedError
