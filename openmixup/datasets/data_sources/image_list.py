import os
import mmcv
import numpy as np
from PIL import Image

from ..registry import DATASOURCES


@DATASOURCES.register_module
class ImageList(object):
    """The implementation for loading any image list file.

    The `ImageList` can load an annotation file or a list of files and merge
    all data records to one list. If data is unlabeled, the gt_label will be
    set -1.

    Args:
        root (str): Path to the dataset.
        list_file (str): Path to the txt list file.
        splitor (str): Splitor between file names and the class id.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='pillow')``.
        return_label (bool): Whether to return the class id.
    """

    CLASSES = None

    def __init__(self,
                 root,
                 list_file,
                 splitor=" ",
                 file_client_args=dict(backend='pillow'),
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

        self.file_client_args = file_client_args
        self.backend = file_client_args.get('backend', 'pillow')
        assert self.backend in \
            ['pillow', 'disk', 'ceph', 'memcached', 'lmdb', 'petrel', 'http'], \
            "Find unsupport file_client_backend={}".format(self.backend)
        if self.backend != 'pillow':
            self.file_client = mmcv.FileClient(**self.file_client_args)
        else:
            self.file_client = None

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if self.backend == 'pillow':
            img = Image.open(self.fns[idx])
            img = img.convert('RGB')
        else:
            img_bytes = self.file_client.get(self.fns[idx])
            img = mmcv.imfrombytes(img_bytes, flag='color')
            if img is None:  # fix bug of loading by cv2
                if self.backend == 'disk':
                    img = Image.open(self.fns[idx])
                    img = img.convert('RGB')
                else:
                    raise ValueError("Fail to load img={}".format(self.fns[idx]))
            else:
                img = Image.fromarray(img.astype(np.uint8))

        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return (img, target)
        else:
            return img
