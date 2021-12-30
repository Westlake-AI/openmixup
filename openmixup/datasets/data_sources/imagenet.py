from ..registry import DATASOURCES
from .image_list import ImageList


@DATASOURCES.register_module
class ImageNet(ImageList):

    def __init__(self, root, list_file, splitor=" ", return_label=True, **kwargs):
        super(ImageNet, self).__init__(
            root, list_file, splitor, return_label)
