import numpy as np
from ..builder import DATASETS
from ..base import BaseDataset
from .image_list import ImageList
from ..registry import DATASOURCES

@DATASOURCES.register_module
class Cars(ImageList):
    def __init__(self,
                 root,
                 list_file,
                 splitor=" ",
                 file_client_args=dict(backend='pillow'),
                 return_label=True):
        super(Cars, self).__init__(
            root, list_file, splitor, file_client_args, return_label)