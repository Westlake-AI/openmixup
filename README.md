# OpenMixup
[![release](https://img.shields.io/badge/release-V0.2.7-%09%2360004F)](https://github.com/Westlake-AI/openmixup/releases)
[![PyPI](https://img.shields.io/pypi/v/openmixup)](https://pypi.org/project/openmixup)
[![docs](https://img.shields.io/badge/docs-latest-%23002FA7)](https://openmixup.readthedocs.io/en/latest/)
[![license](https://img.shields.io/badge/license-Apache--2.0-%23B7A800)](https://github.com/Westlake-AI/openmixup/blob/main/LICENSE)
[![open issues](https://img.shields.io/github/issues-raw/Westlake-AI/openmixup?color=%23FF9600)](https://github.com/Westlake-AI/openmixup/issues)
[![issue resolution](https://img.shields.io/badge/issue%20resolution-1%20d-%23009763)](https://github.com/Westlake-AI/openmixup/issues)

[📘Documentation](https://openmixup.readthedocs.io/en/latest/) |
[🛠️Installation](https://openmixup.readthedocs.io/en/latest/install.html) |
[🚀Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos) |
[👀Awesome Mixup](https://openmixup.readthedocs.io/en/latest/awesome_mixups/Mixup_SL.html) |
[🔍Awesome MIM](https://openmixup.readthedocs.io/en/latest/awesome_selfsup/MIM.html) |
[🆕News](https://openmixup.readthedocs.io/en/latest/changelog.html)

## Introduction

The main branch works with **PyTorch 1.8** (required by some self-supervised methods) or higher (we recommend **PyTorch 1.12**). You can still use **PyTorch 1.6** for supervised classification methods.

`OpenMixup` is an open-source toolbox for supervised, self-, and semi-supervised visual representation learning with mixup based on PyTorch, especially for mixup-related methods. *Recently, `OpenMixup` is on updating to adopt new features and code structures of OpenMMLab 2.0 ([#42](https://github.com/Westlake-AI/openmixup/issues/42)).*

<div align="center">
  <img src="https://user-images.githubusercontent.com/44519745/179018883-a166f0fa-4d51-4ef1-aed1-d0d4643bcffd.jpg" width="100%"/>
</div>

<details open>
<summary>Major Features</summary>

- **Modular Design.**
  OpenMixup follows a similar code architecture of OpenMMLab projects, which decompose the framework into various components, and users can easily build a customized model by combining different modules. OpenMixup is also transplantable to OpenMMLab projects (e.g., [MMPreTrain](https://github.com/open-mmlab/mmpretrain)).

- **All in One.**
  OpenMixup provides popular backbones, mixup methods, semi-supervised, and self-supervised algorithms. Users can perform image classification (CNN & Transformer) and self-supervised pre-training (contrastive and autoregressive) under the same framework.

- **Standard Benchmarks.**
  OpenMixup supports standard benchmarks of image classification, mixup classification, self-supervised evaluation, and provides smooth evaluation on downstream tasks with open-source projects (e.g., object detection and segmentation on [Detectron2](https://github.com/facebookresearch/maskrcnn-benchmark) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)).

- **State-of-the-art Methods.**
  Openmixup provides awesome lists of popular mixup and self-supervised methods. OpenMixup is updating to support more state-of-the-art image classification and self-supervised methods.

</details>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#news-and-updates">News and Updates</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#overview-of-model-zoo">Overview of Model Zoo</a></li>
    <li><a href="#change-log">Change Log</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
    <li><a href="#contributors">Contributors</a></li>
    <li><a href="#contributors-and-contact">Contributors and Contact</a></li>
  </ol>
</details>

## News and Updates

[2023-12-23] `OpenMixup` v0.2.9 is released, updating more features in mixup augmentations, self-supervised learning, and optimizers.

## Installation

OpenMixup is compatible with **Python 3.6/3.7/3.8/3.9** and **PyTorch >= 1.6**. Here are quick installation steps for development:

```shell
conda create -n openmixup python=3.8 pytorch=1.12 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate openmixup
pip install openmim
mim install mmcv-full
git clone https://github.com/Westlake-AI/openmixup.git
cd openmixup
python setup.py develop
```

Please refer to [install.md](docs/en/install.md) for more detailed installation and dataset preparation.

## Getting Started

OpenMixup supports Linux and macOS. It enables easy implementation and extensions of mixup data augmentation methods in existing supervised, self-, and semi-supervised visual recognition models. Please see [get_started.md](docs/en/get_started.md) for the basic usage of OpenMixup.

### Training and Evaluation Scripts

Here, we provide scripts for starting a quick end-to-end training with multiple `GPUs` and the specified `CONFIG_FILE`. 
```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```
For example, you can run the script below to train a ResNet-50 classifier on ImageNet with 4 GPUs:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh configs/classification/imagenet/resnet/resnet50_4xb64_cos_ep100.py 4
```
After training, you can test the trained models with the corresponding evaluation script:
```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${GPUS} ${PATH_TO_MODEL} [optional arguments]
```

### Development

Please see [Tutorials](docs/en/tutorials) for more developing examples and tech details:

- [config files](docs/en/tutorials/0_config.md)
- [add new dataset](docs/en/tutorials/1_new_dataset.md)
- [data pipeline](docs/en/tutorials/2_data_pipeline.md)
- [add new modules](docs/en/tutorials/3_new_module.md)
- [customize schedules](docs/en/tutorials/4_schedule.md)
- [customize runtime](docs/en/tutorials/5_runtime.md)

Downetream Tasks for Self-supervised Learning

- [Classification](docs/en/tutorials/ssl_classification.md)
- [Detection](docs/en/tutorials/ssl_detection.md)
- [Segmentation](docs/en/tutorials/ssl_segmentation.md)

Useful Tools

- [Analysis](docs/en/tutorials/analysis.md)
- [Visualization](docs/en/tutorials/visualization.md)
- [pytorch2onnx](docs/en/tutorials/pytorch2onnx.md)
- [pytorch2torchscript](docs/en/tutorials/pytorch2torchscript.md)

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview of Model Zoo

Please run experiments or find results on each config page. Refer to [Mixup Benchmarks](docs/en/mixup_benchmarks) for benchmarking results of mixup methods. View [Model Zoos Sup](docs/en/model_zoos/Model_Zoo_sup.md) and [Model Zoos SSL](docs/en/model_zoos/Model_Zoo_selfsup.md) for a comprehensive collection of mainstream backbones and self-supervised algorithms. We also provide the paper lists of [Awesome Mixups](docs/en/awesome_mixups) and [Awesome MIM](docs/en/awesome_selfsup/MIM.md) for your reference. Please view config files and links to models at the following config pages. Checkpoints and training logs are on updating!

* Backbone architectures for supervised image classification on ImageNet.

    <details open>
    <summary>Currently supported backbones</summary>

    - [x] [AlexNet](https://dl.acm.org/doi/10.1145/3065386) (NIPS'2012) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/alexnet/)]
    - [x] [VGG](https://arxiv.org/abs/1409.1556) (ICLR'2015) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/vgg/)]
    - [x] [InceptionV3](https://arxiv.org/abs/1512.00567) (CVPR'2016) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/inception_v3/)]
    - [x] [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) (CVPR'2016) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/resnet/)]
    - [x] [ResNeXt](https://arxiv.org/abs/1611.05431) (CVPR'2017) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/resnet/)]
    - [x] [SE-ResNet](https://arxiv.org/abs/1709.01507) (CVPR'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/resnet/)]
    - [x] [SE-ResNeXt](https://arxiv.org/abs/1709.01507) (CVPR'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/resnet/)]
    - [x] [ShuffleNetV1](https://arxiv.org/abs/1807.11164) (CVPR'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/shufflenet_v1/)]
    - [x] [ShuffleNetV2](https://arxiv.org/abs/1807.11164) (ECCV'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/shufflenet_v2/)]
    - [x] [MobileNetV2](https://arxiv.org/abs/1801.04381) (CVPR'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mobilenet_v2/)]
    - [x] [MobileNetV3](https://arxiv.org/abs/1905.02244) (ICCV'2019) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mobilenet_v3/)]
    - [x] [EfficientNet](https://arxiv.org/abs/1905.11946) (ICML'2019) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/efficientnet/)]
    - [x] [EfficientNetV2](https://arxiv.org/abs/2104.00298) (ICML'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/efficientnet_v2/)]
    - [x] [HRNet](https://arxiv.org/abs/1908.07919) (TPAMI'2019) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/hrnet/)]
    - [x] [Res2Net](https://arxiv.org/abs/1904.01169) (ArXiv'2019) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/res2net/)]
    - [x] [CSPNet](https://arxiv.org/abs/1911.11929) (CVPRW'2020) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/cspnet/)]
    - [x] [RegNet](https://arxiv.org/abs/2003.13678) (CVPR'2020) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/regnet/)]
    - [x] [Vision-Transformer](https://arxiv.org/abs/2010.11929) (ICLR'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/vision_transformer/)]
    - [x] [Swin-Transformer](https://arxiv.org/abs/2103.14030) (ICCV'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/swin_transformer/)]
    - [x] [PVT](https://arxiv.org/abs/2102.12122) (ICCV'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/pvt/)]
    - [x] [T2T-ViT](https://arxiv.org/abs/2101.11986) (ICCV'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/t2t_vit/)]
    - [x] [LeViT](https://arxiv.org/abs/2104.01136) (ICCV'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/levit/)]
    - [x] [RepVGG](https://arxiv.org/abs/2101.03697) (CVPR'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/repvgg/)]
    - [x] [DeiT](https://arxiv.org/abs/2012.12877) (ICML'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/deit/)]
    - [x] [MLP-Mixer](https://arxiv.org/abs/2105.01601) (NIPS'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mlp_mixer/)]
    - [x] [Twins](https://proceedings.neurips.cc/paper/2021/hash/4e0928de075538c593fbdabb0c5ef2c3-Abstract.html) (NIPS'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/twins/)]
    - [x] [ConvMixer](https://arxiv.org/abs/2201.09792) (Openreview'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/convmixer/)]
    - [x] [BEiT](https://arxiv.org/abs/2106.08254) (ICLR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/beit/)]
    - [x] [UniFormer](https://arxiv.org/abs/2201.09450) (ICLR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/uniformer/)]
    - [x] [MobileViT](http://arxiv.org/abs/2110.02178) (ICLR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mobilevit/)]
    - [x] [PoolFormer](https://arxiv.org/abs/2111.11418) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/poolformer/)]
    - [x] [ConvNeXt](https://arxiv.org/abs/2201.03545) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/convnext/)]
    - [x] [MViTV2](https://arxiv.org/abs/2112.01526) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mvit/)]
    - [x] [RepMLP](https://arxiv.org/abs/2105.01883) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/repmlp/)]
    - [x] [VAN](https://arxiv.org/abs/2202.09741) (CVMJ'2023) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/van/)]
    - [x] [DeiT-3](https://arxiv.org/abs/2204.07118) (ECCV'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/deit3/)]
    - [x] [LITv2](https://arxiv.org/abs/2205.13213) (NIPS'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/lit_v2/)]
    - [x] [HorNet](https://arxiv.org/abs/2207.14284) (NIPS'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/hornet/)]
    - [x] [DaViT](https://arxiv.org/abs/2204.03645) (ECCV'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/davit/)]
    - [x] [EdgeNeXt](https://arxiv.org/abs/2206.10589) (ECCVW'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/edgenext/)]
    - [x] [EfficientFormer](https://arxiv.org/abs/2206.01191) (ArXiv'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/efficientformer/)]
    - [x] [MogaNet](https://arxiv.org/abs/2211.03295) (ICLR'2024) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/)]
    - [x] [MetaFormer](http://arxiv.org/abs/2210.13452) (ArXiv'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/metaformer/)]
    - [x] [ConvNeXtV2](http://arxiv.org/abs/2301.00808) (ArXiv'2023) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/convnext_v2/)]
    - [x] [CoC](https://arxiv.org/abs/2303.01494) (ICLR'2023) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/context_cluster/)]
    - [x] [MobileOne](http://arxiv.org/abs/2206.04040) (CVPR'2023) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mobileone/)]
    - [x] [VanillaNet](http://arxiv.org/abs/2305.12972) (ArXiv'2023) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/vanillanet/)]
    - [x] [RWKV](https://arxiv.org/abs/2305.13048) (ArXiv'2023) [[config](IP51/openmixup/configs/classification/imagenet/rwkv/)]
    </details>

* Mixup methods for supervised image classification.

    <details open>
    <summary>Currently supported mixup methods</summary>

    - [x] [Mixup](https://arxiv.org/abs/1710.09412) (ICLR'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [CutMix](https://arxiv.org/abs/1905.04899) (ICCV'2019) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [ManifoldMix](https://arxiv.org/abs/1806.05236) (ICML'2019) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [FMix](https://arxiv.org/abs/2002.12047) (ArXiv'2020) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [AttentiveMix](https://arxiv.org/abs/2003.13048) (ICASSP'2020) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [SmoothMix](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.pdf) (CVPRW'2020) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [SaliencyMix](https://arxiv.org/abs/1710.09412) (ICLR'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [PuzzleMix](https://arxiv.org/abs/2009.06962) (ICML'2020) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [SnapMix](https://arxiv.org/abs/2012.04846) (AAAI'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/)]
    - [x] [GridMix](https://www.sciencedirect.com/science/article/pii/S0031320320303976) (Pattern Recognition'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [ResizeMix](https://arxiv.org/abs/2012.11101) (CVMJ'2023) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [AlignMix](https://arxiv.org/abs/2103.15375) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [TransMix](https://arxiv.org/abs/2111.09833) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [AutoMix](https://arxiv.org/abs/2103.13027) (ECCV'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix)]
    - [x] [SAMix](https://arxiv.org/abs/2111.15454) (ArXiv'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix)]
    - [x] [DecoupleMix](https://arxiv.org/abs/2203.10761) (NeurIPS'2023) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/decouple)]
    - [ ] [SMMix](https://arxiv.org/abs/2212.12977) (ICCV'2023) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [AdAutoMix](https://arxiv.org/abs/2312.11954) (ICLR'2024) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/adautomix)]
    </details>

    <details open>
    <summary>Currently supported datasets for mixups</summary>

    - [x] [ImageNet](https://arxiv.org/abs/1409.0575) [[download (1K)](http://www.image-net.org/challenges/LSVRC/2012/)] [[download (21K)](https://image-net.org/data/imagenet21k_resized.tar.gz)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) [[download](https://www.cs.toronto.edu/~kriz/cifar.html)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar10/)]
    - [x] [CIFAR-100](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) [[download](https://www.cs.toronto.edu/~kriz/cifar.html)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/)]
    - [x] [Tiny-ImageNet](https://arxiv.org/abs/1707.08819) [[download](http://cs231n.stanford.edu/tiny-imagenet-200.zip)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/tiny_imagenet/)]
    - [x] [FashionMNIST](https://arxiv.org/abs/1708.07747) [[download](https://github.com/zalandoresearch/fashion-mnist)]
    - [x] [STL-10](http://proceedings.mlr.press/v15/coates11a/coates11a.pdf) [[download](https://cs.stanford.edu/~acoates/stl10/)]
    - [x] [CUB-200-2011](https://resolver.caltech.edu/CaltechAUTHORS:20111026-120541847) [[download](http://www.vision.caltech.edu/datasets/cub_200_2011/)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cub200/)]
    - [x] [FGVC-Aircraft](https://arxiv.org/abs/1306.5151) [[download](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/aircrafts/)]
    - [x] [Stanford-Cars](http://ai.stanford.edu/~jkrause/papers/3drr13.pdf) [[download](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)]
    - [x] [Places205](http://places2.csail.mit.edu/index.html) [[download](http://places.csail.mit.edu/downloadData.html)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/place205/)]
    - [x] [iNaturalist-2017](https://arxiv.org/abs/1707.06642) [[download](https://github.com/visipedia/inat_comp/tree/master/2017)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/inaturalist2017/)]
    - [x] [iNaturalist-2018](https://arxiv.org/abs/1707.06642) [[download](https://github.com/visipedia/inat_comp/tree/master/2018)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/inaturalist2018/)]
    - [x] [AgeDB](https://ieeexplore.ieee.org/document/8014984) [[download](https://ibug.doc.ic.ac.uk/resources/agedb/)] [[download (baidu)](https://pan.baidu.com/s/1XdibVxiGoWf46HLOHKiIyw?pwd=0n6p)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/regression/agedb)]
    - [x] [IMDB-WIKI](https://link.springer.com/article/10.1007/s11263-016-0940-3) [[download (imdb)](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar)] [[download (wiki)](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/regression/imdb_wiki)]
    - [x] [RCFMNIST](https://arxiv.org/abs/2210.05775) [[download](https://github.com/zalandoresearch/fashion-mnist)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/regression/rcfmnist)]
    </details>

* Self-supervised algorithms for visual representation learning.

    <details open>
    <summary>Currently supported self-supervised algorithms</summary>

    - [x] [Relative Location](https://arxiv.org/abs/1505.05192) (ICCV'2015) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/relative_loc/)]
    - [x] [Rotation Prediction](https://arxiv.org/abs/1803.07728) (ICLR'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/rotation_pred/)]
    - [x] [DeepCluster](https://arxiv.org/abs/1807.05520) (ECCV'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/deepcluster/)]
    - [x] [NPID](https://arxiv.org/abs/1805.01978) (CVPR'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/npid/)]
    - [x] [ODC](https://arxiv.org/abs/2006.10645) (CVPR'2020) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/odc/)]
    - [x] [MoCov1](https://arxiv.org/abs/1911.05722) (CVPR'2020) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov1/)]
    - [x] [SimCLR](https://arxiv.org/abs/2002.05709) (ICML'2020) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simclr/)]
    - [x] [MoCoV2](https://arxiv.org/abs/2003.04297) (ArXiv'2020) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov2/)]
    - [x] [BYOL](https://arxiv.org/abs/2006.07733) (NIPS'2020) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/)]
    - [x] [SwAV](https://arxiv.org/abs/2006.09882) (NIPS'2020) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/swav/)]
    - [x] [DenseCL](https://arxiv.org/abs/2011.09157) (CVPR'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/densecl/)]
    - [x] [SimSiam](https://arxiv.org/abs/2011.10566) (CVPR'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/)]
    - [x] [Barlow Twins](https://arxiv.org/abs/2103.03230) (ICML'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/barlowtwins/)]
    - [x] [MoCoV3](https://arxiv.org/abs/2104.02057) (ICCV'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov3/)]
    - [x] [BEiT](https://arxiv.org/abs/2106.08254) (ICLR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/beit/)
    - [x] [MAE](https://arxiv.org/abs/2111.06377) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mae/)]
    - [x] [SimMIM](https://arxiv.org/abs/2111.09886) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/)]
    - [x] [MaskFeat](https://arxiv.org/abs/2112.09133) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/maskfeat/)]
    - [x] [CAE](https://arxiv.org/abs/2202.03026) (ArXiv'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/cae/)]
    - [x] [A2MIM](https://arxiv.org/abs/2205.13943) (ICML'2023) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/)]
    </details>

<p align="right">(<a href="#top">back to top</a>)</p>

## Change Log

Please refer to [changelog.md](docs/en/changelog.md) for more details and release history.

## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.

## Acknowledgement

- OpenMixup is an open-source project for mixup methods and visual representation learning created by researchers in **CAIRI AI Lab**. We encourage researchers interested in backbone architectures, mixup augmentations, and self-supervised learning methods to contribute to OpenMixup!
- This project borrows the architecture design and part of the code from [MMPreTrain](https://github.com/open-mmlab/mmpretrain) and the official implementations of supported algorisms.

<p align="right">(<a href="#top">back to top</a>)</p>

## Citation

If you find this project useful in your research, please consider star `OpenMixup` or cite our [tech report](https://arxiv.org/abs/2209.04851):

```BibTeX
@article{li2022openmixup,
  title = {OpenMixup: A Comprehensive Mixup Benchmark for Visual Classification},
  author = {Siyuan Li and Zedong Wang and Zicheng Liu and Di Wu and Cheng Tan and Stan Z. Li},
  journal = {ArXiv},
  year = {2022},
  volume = {abs/2209.04851}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Contributors and Contact

For help, new features, or reporting bugs associated with OpenMixup, please open a [GitHub issue](https://github.com/Westlake-AI/openmixup/issues) and [pull request](https://github.com/Westlake-AI/openmixup/pulls) with the tag "help wanted" or "enhancement". For now, the direct contributors include: Siyuan Li ([@Lupin1998](https://github.com/Lupin1998)), Zedong Wang ([@Jacky1128](https://github.com/Jacky1128)), and Zicheng Liu ([@pone7](https://github.com/pone7)). We thank all public contributors and contributors from MMPreTrain (MMSelfSup and MMClassification)!

This repo is currently maintained by:

- Siyuan Li (lisiyuan@westlake.edu.cn), Westlake University
- Zedong Wang (wangzedong@westlake.edu.cn), Westlake University
- Zicheng Liu (liuzicheng@westlake.edu.cn), Westlake University

<p align="right">(<a href="#top">back to top</a>)</p>
