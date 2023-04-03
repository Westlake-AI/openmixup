# DeiT

> [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)

## Abstract

Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification. However, these visual transformers are pre-trained with hundreds of millions of images using an expensive infrastructure, thereby limiting their adoption. In this work, we produce a competitive convolution-free transformer by training on Imagenet only. We train them on a single computer in less than 3 days. Our reference vision transformer (86M parameters) achieves top-1 accuracy of 83.1% (single-crop evaluation) on ImageNet with no external data. More importantly, we introduce a teacher-student strategy specific to transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention. We show the interest of this token-based distillation, especially when using a convnet as a teacher. This leads us to report results competitive with convnets for both Imagenet (where we obtain up to 85.2% accuracy) and when transferring to other tasks. We share our code and models.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/179356514-607628eb-7511-4847-99d2-f5f6e6a4560b.png" width="45%"/>
</div>

## Results and models

This page is based on documents in [MMClassification](https://github.com/open-mmlab/mmclassification).

### ImageNet-1k

|   Model   |   Pretrain   | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                Config                                 |                                Download                                 |
| :-------: | :----------: | :-------: | :------: | :-------: | :-------: | :-------------------------------------------------------------------: | :---------------------------------------------------------------------: |
|  DeiT-T   | From scratch |   5.72    |   1.08   |   73.56   |   91.16   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/deit/deit_tiny_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-tiny_pt-4xb256_in1k_20220218-13b382a0.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/deit/deit-tiny_pt-4xb256_in1k_20220218-13b382a0.log.json) |
|  DeiT-T\* | From scratch |   5.72    |   1.08   |   72.20   |   91.10   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/deit/deit_tiny_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-tiny-distilled_3rdparty_pt-4xb256_in1k_20211216-c429839a.pth) |
|  DeiT-S   | From scratch |   22.05   |   4.24   |   79.93   |   95.14   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/deit/deit_small_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/deit/deit-small_pt-4xb256_in1k_20220218-9425b9bb.log.json) |
|  DeiT-S\* | From scratch |   22.05   |   4.24   |   79.90   |   95.10   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/deit/deit_small_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-small-distilled_3rdparty_pt-4xb256_in1k_20211216-4de1d725.pth) |
|  DeiT-B   | From scratch |   86.57   |   16.86  |   81.82   |   95.57   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/deit/deit_base_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-base_pt-16xb64_in1k_20220216-db63c16c.pth)  \| [log](https://download.openmmlab.com/mmclassification/v0/deit/deit-base_pt-16xb64_in1k_20220216-db63c16c.log.json) |
|  DeiT-B\* | From scratch |   86.57   |   16.86  |   81.80   |   95.60   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/deit/deit_base_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-base_3rdparty_pt-16xb64_in1k_20211124-6f40c188.pth) |
|  DeiT-B distilled\* | From scratch |   86.57   |  16.86   |   83.33   |   96.49   |    [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/deit/deit_base_8xb128_ep300.py)    | [model](https://download.openmmlab.com/mmclassification/v0/deit/deit-base-distilled_3rdparty_pt-16xb64_in1k_20211216-42891296.pth) |

We follow the original training setting provided by the [official repo](https://github.com/facebookresearch/deit) and reproduce the performance of 300-epoch training from scratch without distillation. *Note that this repo does not support the distillation loss in DeiT. Models with * are provided by the [official repo](https://github.com/facebookresearch/deit).*

## Citation

```bibtex
@InProceedings{icml2021deit,
  title =     {Training data-efficient image transformers &amp; distillation through attention},
  author =    {Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and Jegou, Herve},
  booktitle = {International Conference on Machine Learning},
  pages =     {10347--10357},
  year =      {2021},
  volume =    {139},
  month =     {July}
}
```
