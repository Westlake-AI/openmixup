# Tutorial 1: Adding New Dataset

## Customize datasets by reorganizing data

### Reorganize dataset to existing format

The simplest way is to convert your dataset to existing dataset formats (`ImageList` or `ImageNet`) using with a meta file.

For training, it differentiates classes by folders. The directory of training data is as follows:

```
imagenet
├── ...
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── ...
│   ├── ...
│   ├── n15075141
│   │   ├── n15075141_999.JPEG
│   │   ├── n15075141_9993.JPEG
│   │   ├── ...
```

For validation, we provide a annotation list. Each line of the list contrains a filename and its corresponding ground-truth labels. The format is as follows:

```
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
ILSVRC2012_val_00000003.JPEG 230
ILSVRC2012_val_00000004.JPEG 809
ILSVRC2012_val_00000005.JPEG 516
```

Note: The value of ground-truth labels should fall in range `[0, num_classes - 1]`. Please refer to [INSTALL](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/install.md) [meta files](https://github.com/Westlake-AI/openmixup/releases/download/dataset/meta.zip) for examples.

### An example of customized dataset

You can write a new Dataset class inherited from `ImageList`, and overwrite `get_sample(self)`. Typically, this function returns a list (`img` and `gt_label`) or a raw image. Assume we are going to implement a `Filelist` dataset, which takes filelists for both training and testing. The format of annotation list is as follows:

```
000001.jpg 0
000002.jpg 1
```

We can create a new dataset in `openmixup/datasets/data_sources/filelist.py` to load the data.

```python
import mmcv
import numpy as np

from ..registry import DATASOURCES
from .image_list import ImageList


@DATASOURCES.register_module
class Filelist(ImageList):

    def get_sample(self, idx):
        # TODO: load samples from the idx
        img_bytes = self.file_client.get(self.fns[idx])
        img = mmcv.imfrombytes(img_bytes, flag='color')
        target = self.labels[idx]
        return (img, target)
```

And add this dataset class in `mmcls/datasets/__init__.py`

```python
from .base_dataset import BaseDataset
...
from .filelist import Filelist

__all__ = [
    'BaseDataset', ... ,'Filelist'
]
```

Then in the config, to use `Filelist` you can modify the config as the following

```python
train = dict(
    type='Filelist',
    ann_file = 'image_list.txt',
    pipeline=train_pipeline
)
```

## Customize datasets by mixing dataset

OpenMixup also supports to mix dataset for training. Currently it supports to concat and repeat datasets.

### Repeat dataset

We use `RepeatDataset` as wrapper to repeat the dataset. For example, suppose the original dataset is `Dataset_A`, to repeat it, the config looks like the following

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

<p align="right">(<a href="#top">back to top</a>)</p>
