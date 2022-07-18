# OpenMixup
[📘Documentation](https://openmixup.readthedocs.io/en/latest/) |
[🛠️Installation](https://openmixup.readthedocs.io/en/latest/install.html) |
[🚀Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos) |
[👀Awesome Mixup](https://openmixup.readthedocs.io/en/latest/awesome_mixups/Mixup_SL.html) |
[🆕News](https://openmixup.readthedocs.io/en/latest/changelog.html)

## Introduction

The main branch works with **PyTorch 1.8** (required by some self-supervised methods) or higher (we recommend **PyTorch 1.10**). You can still use **PyTorch 1.6** for supervised classification methods.

`OpenMixup` is an open-source toolbox for supervised, self-, and semi-supervised visual representation learning with mixup based on PyTorch, especially for mixup-related methods.

<div align="center">
  <img src="https://user-images.githubusercontent.com/44519745/179018883-a166f0fa-4d51-4ef1-aed1-d0d4643bcffd.jpg" width="100%"/>
</div>

<details open>
<summary>Major Features</summary>

- **Modular Design.**
  OpenMixup follows a similar code architecture of OpenMMLab projects, which decompose the framework into various components, and users can easily build a customized model by combining different modules. OpenMixup is also transplatable to OpenMMLab projects (e.g., [MMSelfSup](https://github.com/open-mmlab/mmselfsup)).

- **All in One.**
  OpenMixup provides popular backbones, mixup methods, semi-supervised, and self-supervised algorithms. Users can perform image classification (CNN & Transformer) and self-supervised pre-training (contrastive and autoregressive) under the same setting.

- **Standard Benchmarks.**
  OpenMixup supports standard benchmarks of image classification, mixup classification, self-supervised evaluation, and provides smooth evaluation on downstream tasks with open-source projects (e.g., object detection and segmentation on [Detectron2](https://github.com/facebookresearch/maskrcnn-benchmark) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)).

</details>

## What's New

[2020-07-07] `OpenMixup` v0.2.4 is released (issue [#7](https://github.com/Westlake-AI/openmixup/issues/7)), which fixs bugs of weight initialization and fine-tuning, updates docs, etc.

## Installation

There are quick installation steps for develepment:

```shell
conda create -n openmixup python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate openmixup
pip3 install openmim
mim install mmcv-full
git clone https://github.com/Westlake-AI/openmixup.git
cd openmixup
python setup.py develop
```

Please refer to [install.md](docs/en/install.md) for more detailed installation and dataset preparation.

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of OpenMixup. You can start a multiple GPUs training with `CONFIG_FILE` using the following script. An example, 
```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```
Please Then, see [Tutorials](docs/en/tutorials) for more tech details:

- [config files](docs/en/tutorials/0_config.md)
- [add new dataset](docs/en/tutorials/1_new_dataset.md)
- [data pipeline](docs/en/tutorials/2_data_pipeline.md)
- [add new modules](docs/en/tutorials/3_new_module.md)
- [customize schedules](docs/en/tutorials/4_schedule.md)
- [customize runtime](docs/en/tutorials/5_runtime.md)
- [benchmarks](docs/en/tutorials/6_benchmarks.md)

## Overview of Model Zoo

Please refer to [Model Zoos](docs/en/model_zoos) for various backbones, mixup methods, and self-supervised algorithms. We also provide the paper lists of [Awesome Mixups](docs/en/awesome_mixups) for your reference. Checkpoints and traning logs will be updated soon!

* Backbone architectures for supervised image classification on ImageNet.

    <details open>
    <summary>Currently supported backbones</summary>

    - [x] [VGG](https://arxiv.org/abs/1409.1556) (ICLR'2015) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/vgg/)]
    - [x] [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) (CVPR'2016) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/resnet/)]
    - [x] [ResNeXt](https://arxiv.org/abs/1611.05431) (CVPR'2017) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/resnet/)]
    - [x] [SE-ResNet](https://arxiv.org/abs/1709.01507) (CVPR'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/resnet/)]
    - [x] [SE-ResNeXt](https://arxiv.org/abs/1709.01507) (CVPR'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/resnet/)]
    - [x] [ShuffleNetV2](https://arxiv.org/abs/1807.11164) (ECCV'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/shufflenet/)]
    - [x] [MobileNetV2](https://arxiv.org/abs/1801.04381) (CVPR'2018) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mobilenet_v2/)]
    - [x] [MobileNetV3](https://arxiv.org/abs/1905.02244) (ICCV'2019)
    - [x] [EfficientNet](https://arxiv.org/abs/1905.11946) (ICML'2019) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/efficientnet/)]
    - [x] [Swin-Transformer](https://arxiv.org/abs/2103.14030) (ICCV'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/swin_transformer/)]
    - [x] [RepVGG](https://arxiv.org/abs/2101.03697) (CVPR'2021)
    - [x] [Vision-Transformer](https://arxiv.org/pdf/2010.11929.pdf) (ICLR'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/vision_transformer/)]
    - [x] [MLP-Mixer](https://arxiv.org/abs/2105.01601) (NIPS'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mlp_mixer/)]
    - [x] [DeiT](https://arxiv.org/abs/2012.12877) (ICML'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/deit/)]
    - [x] [ConvMixer](https://arxiv.org/abs/2201.09792) (Openreview'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/convmixer/)]
    - [x] [UniFormer](https://arxiv.org/abs/2201.09450) (ICLR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/uniformer/)]
    - [x] [PoolFormer](https://arxiv.org/abs/2111.11418) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/poolformer/)]
    - [x] [ConvNeXt](https://arxiv.org/abs/2201.03545) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/convnext/)]
    - [x] [VAN](https://arxiv.org/abs/2202.09741) (ArXiv'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/van/)]
    - [x] [LITv2](https://arxiv.org/abs/2205.13213) (ArXiv'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/lit_v2/)]
    </details>

* Mixup methods for supervised image classification.

    <details open>
    <summary>Currently supported mixup methods</summary>

    - [x] [Mixup](https://arxiv.org/abs/1710.09412) (ICLR'2018)
    - [x] [CutMix](https://arxiv.org/abs/1905.04899) (ICCV'2019)
    - [x] [ManifoldMix](https://arxiv.org/abs/1806.05236) (ICML'2019)
    - [x] [FMix](https://arxiv.org/abs/2002.12047) (ArXiv'2020)
    - [x] [AttentiveMix](https://arxiv.org/abs/2003.13048) (ICASSP'2020)
    - [x] [SmoothMix](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.pdf) (CVPRW'2020)
    - [x] [SaliencyMix](https://arxiv.org/abs/1710.09412) (ICLR'2021)
    - [x] [PuzzleMix](https://arxiv.org/abs/2009.06962) (ICML'2020)
    - [x] [GridMix](https://www.sciencedirect.com/science/article/pii/S0031320320303976) (Pattern Recognition'2021)
    - [x] [ResizeMix](https://arxiv.org/abs/2012.11101) (ArXiv'2020)
    - [x] [AutoMix](https://arxiv.org/abs/2103.13027) (ECCV'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix)]
    - [x] [SAMix](https://arxiv.org/abs/2111.15454) (ArXiv'2021) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix)]
    </details>

    <details open>
    <summary>Currently supported datasets for mixups</summary>

    - [x] [ImageNet](https://dl.acm.org/doi/10.1145/3065386) [[download](http://www.image-net.org/challenges/LSVRC/2012/)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/)]
    - [x] [CIFAR-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) [[download](https://www.cs.toronto.edu/~kriz/cifar.html)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar10/)]
    - [x] [CIFAR-100](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) [[download](https://www.cs.toronto.edu/~kriz/cifar.html)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/)]
    - [x] [Tiny-ImageNet](https://arxiv.org/abs/1707.08819) [[download](https://www.kaggle.com/c/tiny-imagenet)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/tiny_imagenet/)]
    - [x] [CUB-200-2011](https://resolver.caltech.edu/CaltechAUTHORS:20111026-120541847) [[download](http://www.vision.caltech.edu/datasets/cub_200_2011/)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cub200/)]
    - [x] [FGVC-Aircraft](https://arxiv.org/abs/1306.5151) [[download](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/aircrafts/)]
    - [x] [StandfoldCars](http://ai.stanford.edu/~jkrause/papers/3drr13.pdf) [[download](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)]
    - [x] [Place205](http://places2.csail.mit.edu/index.html) [[download](http://places.csail.mit.edu/downloadData.html)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/place205/)]
    - [x] [iNaturalist-2017/2018](https://arxiv.org/abs/1707.06642) [[download](https://github.com/visipedia/inat_comp)] [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/inaturalist2017/)]
    </details>

* Self-supervised algorithms for visual representation.

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
    - [x] [MAE](https://arxiv.org/abs/2111.06377) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mae/)]
    - [x] [SimMIM](https://arxiv.org/abs/2111.09886) (CVPR'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/)]
    - [x] [CAE](https://arxiv.org/abs/2202.03026) (ArXiv'2022)
    - [x] [A2MIM](https://arxiv.org/abs/2205.13943) (ArXiv'2022) [[config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/)]
    </details>

## Change Log

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

- OpenMixup is an open-source project for mixup methods created by researchers in **CAIRI AI LAB**. We encourage researchers interested in visual representation learning and mixup methods to contribute to OpenMixup!
- This repo borrows the architecture design and part of the code from [MMSelfSup](https://github.com/open-mmlab/mmselfsup) and [MMClassification](https://github.com/open-mmlab/mmclassification).

## Citation

If you find this project useful in your research, please consider cite our repo:

```BibTeX
@misc{2022openmixup,
    title={{OpenMixup}: Open Mixup Toolbox and Benchmark for Visual Representation Learning},
    author={Li, Siyuan and Liu, Zichen and Wu, Di and Stan Z. Li},
    howpublished = {\url{https://github.com/Westlake-AI/openmixup}},
    year={2022}
}
```

## Contributors

For now, the direct contributors include: Siyuan Li ([@Lupin1998](https://github.com/Lupin1998)), Zicheng Liu ([@pone7](https://github.com/pone7)), and Di Wu ([@wudi-bu](https://github.com/wudi-bu)). We thank contributors from MMSelfSup and MMClassification.

## Contact

This repo is currently maintained by Siyuan Li (lisiyuan@westlake.edu.cn) and Zicheng Liu (liuzicheng@westlake.edu.cn).
