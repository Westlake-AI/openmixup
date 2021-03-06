import os
from PIL import Image
import mmcv

from ..registry import DATASOURCES


@DATASOURCES.register_module
class ImageList(object):
    """The implementation for loading any image list file.

    The `ImageList` can load an annotation file or a list of files and merge
    all data records to one list. If data is unlabeled, the gt_label will be
    set -1.
    """

    CLASSES = None

    def __init__(self,
                 root,
                 list_file,
                 splitor=" ",
                 backend='pillow',
                 return_label=True):
        with open(list_file, 'r') as fp:
            lines = fp.readlines()
        fp.close()
        assert splitor in [" ", ",", ";"]
        self.has_labels = len(lines[0].split(splitor)) == 2
        self.return_label = return_label
        if self.has_labels:
            self.fns, self.labels = zip(*[l.strip().split(splitor) for l in lines])
            self.labels = [int(l) for l in self.labels]
        else:
            # assert self.return_label is False
            self.labels = None
            self.fns = [l.strip() for l in lines]
        self.fns = [os.path.join(root, fn) for fn in self.fns]
        self.backend = backend
        if self.backend == 'cv2':
            self.file_client = mmcv.FileClient(backend='disk')

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if self.backend == 'pillow':
            img = Image.open(self.fns[idx])
            img = img.convert('RGB')
        else:
            img_bytes = self.file_client.get(self.fns[idx])
            img = mmcv.imfrombytes(img_bytes, flag='color')
        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return img, target
        else:
            return img
