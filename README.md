# OpenMixup
[![release](https://img.shields.io/badge/release-V0.2.7-%09%2360004F)](https://github.com/Westlake-AI/openmixup/releases)
[![PyPI](https://img.shields.io/pypi/v/openmixup)](https://pypi.org/project/openmixup)
[![arxiv](https://img.shields.io/badge/arXiv-2209.04851-b31b1b.svg?style=flat)](https://arxiv.org/abs/2209.04851)
[![docs](https://img.shields.io/badge/docs-latest-%23002FA7)](https://openmixup.readthedocs.io/en/latest/)
[![license](https://img.shields.io/badge/license-Apache--2.0-%23B7A800)](https://github.com/Westlake-AI/openmixup/blob/main/LICENSE)
[![open issues](https://img.shields.io/github/issues-raw/Westlake-AI/openmixup?color=%23009763)](https://github.com/Westlake-AI/openmixup/issues)
<!-- [![issue resolution](https://img.shields.io/badge/issue%20resolution-1%20d-%23009763)](https://github.com/Westlake-AI/openmixup/issues) -->

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

[2025-03-19] `OpenMixup` v0.2.10 is released, supporting **PyTorch >= 2.0** and more mixup augmentations and networks.

## Installation

OpenMixup is compatible with **Python 3.6/3.7/3.8/3.9** and **PyTorch >= 1.6**. Here are quick installations for installation in the development mode:

```shell
conda create -n openmixup python=3.8 pytorch=1.12 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate openmixup
pip install openmim
mim install mmcv-full
git clone https://github.com/Westlake-AI/openmixup.git
cd openmixup
python setup.py develop
```

<details>
  <summary>Installation with PyTorch 2.x requiring different processes.</summary>

  ```bash
  conda create -n openmixup python=3.9
  conda activate openmixup
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
  pip install https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv_full-1.7.2-cp39-cp39-manylinux1_x86_64.whl
  git clone https://github.com/Westlake-AI/openmixup.git
  cd openmixup
  pip install -r requirements/runtime.txt
  python setup.py develop
  ```
</details>

Fore more detailed installation and dataset preparation, please refer to [install.md](docs/en/install.md).

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

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Supported Backbone Architectures</b>
      </td>
      <td>
        <b>Mixup Data Augmentations</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <li><a href="https://dl.acm.org/doi/10.1145/3065386">AlexNet</a> (NeurIPS'2012) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/alexnet/">config</a></li>
        <li><a href="https://arxiv.org/abs/1409.1556">VGG</a> (ICLR'2015) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/vgg/">config</a></li>
        <li><a href="https://arxiv.org/abs/1512.00567">InceptionV3</a> (CVPR'2016) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/inception_v3/">config</a></li>
        <li><a href="https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">ResNet</a> (CVPR'2016) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/resnet/">config</a></li>
        <li><a href="https://arxiv.org/abs/1611.05431">ResNeXt</a> (CVPR'2017) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/resnet/">config</a></li>
        <li><a href="https://arxiv.org/abs/1709.01507">SE-ResNet</a> (CVPR'2018) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/resnet/">config</a></li>
        <li><a href="https://arxiv.org/abs/1709.01507">SE-ResNeXt</a> (CVPR'2018) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/resnet/">config</a></li>
        <li><a href="https://arxiv.org/abs/1807.11164">ShuffleNetV1</a> (CVPR'2018) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/shufflenet_v1/">config</a></li>
        <li><a href="https://arxiv.org/abs/1807.11164">ShuffleNetV2</a> (ECCV'2018) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/shufflenet_v2/">config</a></li>
        <li><a href="https://arxiv.org/abs/1801.04381">MobileNetV2</a> (CVPR'2018) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mobilenet_v2/">config</a></li>
        <li><a href="https://arxiv.org/abs/1905.02244">MobileNetV3</a> (ICCV'2019) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mobilenet_v3/">config</a></li>
        <li><a href="https://arxiv.org/abs/1905.11946">EfficientNet</a> (ICML'2019) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/efficientnet/">config</a></li>
        <li><a href="https://arxiv.org/abs/2104.00298">EfficientNetV2</a> (ICML'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/efficientnet_v2/">config</a></li>
        <li><a href="https://arxiv.org/abs/1908.07919">HRNet</a> (TPAMI'2019) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/hrnet/">config</a></li>
        <li><a href="https://arxiv.org/abs/1904.01169">Res2Net</a> (ArXiv'2019) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/res2net/">config</a></li>
        <li><a href="https://arxiv.org/abs/1911.11929">CSPNet</a> (CVPRW'2020) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/cspnet/">config</a></li>
        <li><a href="https://arxiv.org/abs/2003.13678">RegNet</a> (CVPR'2020) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/regnet/">config</a></li>
        <li><a href="https://arxiv.org/abs/2010.11929">Vision-Transformer</a> (ICLR'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/vision_transformer/">config</a></li>
        <li><a href="https://arxiv.org/abs/2103.14030">Swin-Transformer</a> (ICCV'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/swin_transformer/">config</a></li>
        <li><a href="https://arxiv.org/abs/2102.12122">PVT</a> (ICCV'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/pvt/">config</a></li>
        <li><a href="https://arxiv.org/abs/2101.11986">T2T-ViT</a> (ICCV'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/t2t_vit/">config</a></li>
        <li><a href="https://arxiv.org/abs/2104.01136">LeViT</a> (ICCV'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/levit/">config</a></li>
        <li><a href="https://arxiv.org/abs/2101.03697">RepVGG</a> (CVPR'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/repvgg/">config</a></li>
        <li><a href="https://arxiv.org/abs/2012.12877">DeiT</a> (ICML'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/deit/">config</a></li>
        <li><a href="https://arxiv.org/abs/2105.01601">MLP-Mixer</a> (NeurIPS'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mlp_mixer/">config</a></li>
        <li><a href="https://proceedings.neurips.cc/paper/2021/hash/4e0928de075538c593fbdabb0c5ef2c3-Abstract.html">Twins</a> (NeurIPS'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/twins/">config</a></li>
        <li><a href="https://arxiv.org/abs/2201.09792">ConvMixer</a> (TMLR'2023) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/convmixer/">config</a></li>
        <li><a href="https://arxiv.org/abs/2106.08254">BEiT</a> (ICLR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/beit/">config</a></li>
        <li><a href="https://arxiv.org/abs/2201.09450">UniFormer</a> (ICLR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/uniformer/">config</a></li>
        <li><a href="http://arxiv.org/abs/2110.02178">MobileViT</a> (ICLR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mobilevit/">config</a></li>
        <li><a href="https://arxiv.org/abs/2111.11418">PoolFormer</a> (CVPR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/poolformer/">config</a></li>
        <li><a href="https://arxiv.org/abs/2201.03545">ConvNeXt</a> (CVPR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/convnext/">config</a></li>
        <li><a href="https://arxiv.org/abs/2112.01526">MViTV2</a> (CVPR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mvit/">config</a></li>
        <li><a href="https://arxiv.org/abs/2105.01883">RepMLP</a> (CVPR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/repmlp/">config</a></li>
        <li><a href="https://arxiv.org/abs/2202.09741">VAN</a> (CVMJ'2023) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/van/">config</a></li>
        <li><a href="https://arxiv.org/abs/2204.07118">DeiT-3</a> (ECCV'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/deit3/">config</a></li>
        <li><a href="https://arxiv.org/abs/2205.13213">LITv2</a> (NeurIPS'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/lit_v2/">config</a></li>
        <li><a href="https://arxiv.org/abs/2207.14284">HorNet</a> (NeurIPS'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/hornet/">config</a></li>
        <li><a href="https://arxiv.org/abs/2204.03645">DaViT</a> (ECCV'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/davit/">config</a></li>
        <li><a href="https://arxiv.org/abs/2206.10589">EdgeNeXt</a> (ECCVW'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/edgenext/">config</a></li>
        <li><a href="https://arxiv.org/abs/2206.01191">EfficientFormer</a> (NeurIPS'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/efficientformer/">config</a></li>
        <li><a href="https://arxiv.org/abs/2211.03295">MogaNet</a> (ICLR'2024) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/">config</a></li>
        <li><a href="http://arxiv.org/abs/2210.13452">MetaFormer</a> (TPAMI'2024) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/metaformer/">config</a></li>
        <li><a href="http://arxiv.org/abs/2301.00808">ConvNeXtV2</a> (CVPR'2023) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/convnext_v2/">config</a></li>
        <li><a href="https://arxiv.org/abs/2303.01494">CoC</a> (ICLR'2023) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/context_cluster/">config</a></li>
        <li><a href="http://arxiv.org/abs/2206.04040">MobileOne</a> (CVPR'2023) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mobileone/">config</a></li>
        <li><a href="http://arxiv.org/abs/2305.12972">VanillaNet</a> (NeurIPS'2023) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/vanillanet/">config</a></li>
        <li><a href="https://arxiv.org/abs/2305.13048">RWKV</a> (ArXiv'2023) <a href="IP51/openmixup/configs/classification/imagenet/rwkv/">config</a></li>
        <li><a href="https://arxiv.org/abs/2311.15599">UniRepLKNet</a> (CVPR'2024) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/unireplknet/">config</a></li>
        <li><a href="https://arxiv.org/abs/2311.17132">TransNeXt</a> (CVPR'2024) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/transnext/">config</a></li>
        <li><a href="https://arxiv.org/abs/2403.19967">StarNet</a> (CVPR'2024) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/starnet/">config</a></li>
        </ul>
      </td>
      <td>
        <ul>
        <li><a href="https://arxiv.org/abs/1710.09412">Mixup</a> (ICLR'2018) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://arxiv.org/abs/1905.04899">CutMix</a> (ICCV'2019) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://arxiv.org/abs/1806.05236">ManifoldMix</a> (ICML'2019) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://arxiv.org/abs/2002.12047">FMix</a> (ArXiv'2020) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://arxiv.org/abs/2003.13048">AttentiveMix</a> (ICASSP'2020) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://openaccess.thecvf.com/content_CVPRW_2020/papers/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.pdf">SmoothMix</a> (CVPRW'2020) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://arxiv.org/abs/1710.09412">SaliencyMix</a> (ICLR'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://arxiv.org/abs/2009.06962">PuzzleMix</a> (ICML'2020) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://arxiv.org/abs/2012.04846">SnapMix</a> (AAAI'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/">config</a></li>
        <li><a href="https://www.sciencedirect.com/science/article/pii/S0031320320303976">GridMix</a> (Pattern Recognition'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://arxiv.org/abs/2012.11101">ResizeMix</a> (CVMJ'2023) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://arxiv.org/abs/2103.15375">AlignMix</a> (CVPR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://arxiv.org/abs/2111.09833">TransMix</a> (CVPR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://arxiv.org/abs/2103.13027">AutoMix</a> (ECCV'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix">config</a></li>
        <li><a href="https://arxiv.org/abs/2111.15454">SAMix</a> (ArXiv'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix">config</a></li>
        <li><a href="https://arxiv.org/abs/2207.08409">TokenMix</a> (ECCV'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/vits">config</a></li>
        <li><a href="https://arxiv.org/abs/2304.12043">MixPro</a> (ICLR'2023) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/vits">config</a></li>
        <li><a href="https://arxiv.org/abs/2203.10761">DecoupleMix</a> (NeurIPS'2023) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/decouple">config</a></li>
        <li><a href="https://arxiv.org/abs/2212.12977">SMMix</a> (ICCV'2023) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/vits">config</a></li>
        <li><a href="https://arxiv.org/abs/2210.06455">TLA</a> (ICCV'2023) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/vits">config</a></li>
        <li><a href="https://arxiv.org/abs/2312.11954">AdAutoMix</a> (ICLR'2024) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/adautomix">config</a></li>
        <li><a href="https://arxiv.org/abs/2407.07805">SUMix</a> (ECCV'2024) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/vits">config</a></li>
        </ul>
      </td>
  </tbody>
</table>


<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Self-supervised Learning Algorithms</b>
      </td>
      <td>
        <b>Supported Datasets</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <li><a href="https://arxiv.org/abs/1505.05192">Relative Location</a> (ICCV'2015) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/relative_loc/">config</a></li>
        <li><a href="https://arxiv.org/abs/1803.07728">Rotation Prediction</a> (ICLR'2018) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/rotation_pred/">config</a></li>
        <li><a href="https://arxiv.org/abs/1807.05520">DeepCluster</a> (ECCV'2018) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/deepcluster/">config</a></li>
        <li><a href="https://arxiv.org/abs/1805.01978">NPID</a> (CVPR'2018) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/npid/">config</a></li>
        <li><a href="https://arxiv.org/abs/2006.10645">ODC</a> (CVPR'2020) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/odc/">config</a></li>
        <li><a href="https://arxiv.org/abs/1911.05722">MoCov1</a> (CVPR'2020) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov1/">config</a></li>
        <li><a href="https://arxiv.org/abs/2002.05709">SimCLR</a> (ICML'2020) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simclr/">config</a></li>
        <li><a href="https://arxiv.org/abs/2003.04297">MoCoV2</a> (ArXiv'2020) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov2/">config</a></li>
        <li><a href="https://arxiv.org/abs/2006.07733">BYOL</a> (NeurIPS'2020) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/">config</a></li>
        <li><a href="https://arxiv.org/abs/2006.09882">SwAV</a> (NeurIPS'2020) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/swav/">config</a></li>
        <li><a href="https://arxiv.org/abs/2011.09157">DenseCL</a> (CVPR'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/densecl/">config</a></li>
        <li><a href="https://arxiv.org/abs/2011.10566">SimSiam</a> (CVPR'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/">config</a></li>
        <li><a href="https://arxiv.org/abs/2103.03230">Barlow Twins</a> (ICML'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/barlowtwins/">config</a></li>
        <li><a href="https://arxiv.org/abs/2104.02057">MoCoV3</a> (ICCV'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov3/">config</a></li>
        <li><a href="https://arxiv.org/abs/2104.14294">DINO</a> (ICCV'2021) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/dino/">config</a></li>
        <li><a href="https://arxiv.org/abs/2106.08254">BEiT</a> (ICLR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/beit/">config</a></li>
        <li><a href="https://arxiv.org/abs/2111.06377">MAE</a> (CVPR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mae/">config</a></li>
        <li><a href="https://arxiv.org/abs/2111.09886">SimMIM</a> (CVPR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/">config</a></li>
        <li><a href="https://arxiv.org/abs/2112.09133">MaskFeat</a> (CVPR'2022) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/maskfeat/">config</a></li>
        <li><a href="https://arxiv.org/abs/2202.03026">CAE</a> (IJCV'2024) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/cae/">config</a></li>
        <li><a href="https://arxiv.org/abs/2205.13943">A2MIM</a> (ICML'2023) <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/">config</a></li>
        </ul>
      </td>
      <td>
        <ul>
        <li><a href="https://arxiv.org/abs/1409.0575">ImageNet</a> [<a href="http://www.image-net.org/challenges/LSVRC/2012/">download (1K)</a>] [<a href="https://image-net.org/data/imagenet21k_resized.tar.gz">download (21K)</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/">config</a></li>
        <li><a href="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf">CIFAR-10 [<a href="https://www.cs.toronto.edu/~kriz/cifar.html">download</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar10/">config</a></li>
        <li><a href="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf">CIFAR-100</a> [<a href="https://www.cs.toronto.edu/~kriz/cifar.html">download</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/">config</a></li>
        <li><a href="https://arxiv.org/abs/1707.08819">Tiny-ImageNet [<a href="http://cs231n.stanford.edu/tiny-imagenet-200.zip">download</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/tiny_imagenet/">config</a></li>
        <li><a href="https://arxiv.org/abs/1708.07747">FashionMNIST</a> [<a href="https://github.com/zalandoresearch/fashion-mnist">download</a>]</li>
        <li><a href="http://proceedings.mlr.press/v15/coates11a/coates11a.pdf">STL-10 [<a href="https://cs.stanford.edu/~acoates/stl10/">download</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov2/stl10">config</a></li>
        <li><a href="https://resolver.caltech.edu/CaltechAUTHORS:20111026-120541847">CUB-200-2011</a> [<a href="http://www.vision.caltech.edu/datasets/cub_200_2011/">download</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cub200/">config</a></li>
        <li><a href="https://arxiv.org/abs/1306.5151">FGVC-Aircraft [<a href="https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/">download</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/aircrafts/">config</a></li>
        <li><a href="http://ai.stanford.edu/~jkrause/papers/3drr13.pdf">Stanford-Cars</a> [<a href="http://ai.stanford.edu/~jkrause/cars/car_dataset.html">download</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cars/">config</a></li>
        <li><a href="http://places2.csail.mit.edu/index.html">Places205 [<a href="http://places.csail.mit.edu/downloadData.html">download</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/place205/">config</a></li>
        <li><a href="https://arxiv.org/abs/1707.06642">iNaturalist-2017</a> [<a href="https://github.com/visipedia/inat_comp/tree/master/2017">download</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/inaturalist2017/">config</a></li>
        <li><a href="https://arxiv.org/abs/1707.06642">iNaturalist-2018</a> [<a href="https://github.com/visipedia/inat_comp/tree/master/2018">download</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/inaturalist2018/">config</a></li>
        <li><a href="https://ieeexplore.ieee.org/document/8014984">AgeDB</a> [<a href="https://ibug.doc.ic.ac.uk/resources/agedb/">download</a>] [<a href="https://pan.baidu.com/s/1XdibVxiGoWf46HLOHKiIyw?pwd=0n6p">download (baidu)</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/regression/agedb">config</a></li>
        <li><a href="https://link.springer.com/article/10.1007/s11263-016-0940-3">IMDB-WIKI</a> [<a href="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar">download (imdb)</a>] [<a href="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar">download (wiki)</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/regression/imdb_wiki">config</a></li>
        <li><a href="https://arxiv.org/abs/2210.05775">RCFMNIST</a> [<a href="https://github.com/zalandoresearch/fashion-mnist">download</a>] <a href="https://github.com/Westlake-AI/openmixup/tree/main/configs/regression/rcfmnist">config</a></li>
        </ul>
      </td>
  </tbody>
</table>

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
