from abc import ABCMeta, abstractmethod

import numpy as np
from openmixup.utils import build_from_cfg
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .registry import PIPELINES
from .builder import build_datasource


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_source (dict): Data source defined in
            `openselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `oenselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        self.data_source = build_datasource(data_source)
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.prefetch = prefetch
        self.CLASSES = self.data_source.CLASSES
        self.targets = self.data_source.labels
        if isinstance(self.targets, list):
            self.targets = np.array(self.targets)

    def __len__(self):
        return self.data_source.get_length()

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def evaluate(self, scores, keyword, logger=None, **kwargs):
        """The evaluation function to predictions."""
        pass
