# OpenMixup
[📘Documentation](https://openmixup.readthedocs.io/en/latest/) |
[🛠️Installation](https://openmixup.readthedocs.io/en/latest/install.html) |
[🚀Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos) |
[👀Awesome Mixup](https://openmixup.readthedocs.io/en/latest/awesome_mixups/Mixup_SL.html) |
[🆕News](https://openmixup.readthedocs.io/en/latest/changelog.html)

**News**

* OpenMixup v0.2.4 is released, which fixs bugs [#7](https://github.com/Westlake-AI/openmixup/issues/7), e.g., weight initialization, fine-tuning.
* OpenMixup v0.2.3 is released, which supports new self-supervised and mixup methods (e.g., [A2MIM](https://arxiv.org/abs/2205.13943)) and backbones (e.g., [UniFormer](https://arxiv.org/abs/2201.09450)), update the [online document](https://westlake-ai.github.io/openmixup/) and config files, and adds new features as [#6](https://github.com/Westlake-AI/openmixup/issues/6).

## Introduction

The main branch works with **PyTorch 1.8** (required by some self-supervised methods) or higher (we recommend **PyTorch 1.10**). You can still use **PyTorch 1.6** for supervised classification methods.

`OpenMixup` is an open-source toolbox for supervised, self-, and semi-supervised visual representation learning with mixup based on PyTorch, especially for mixup-related methods.

### What does this repo do?

Learning discriminative visual representation efficiently that facilitates downstream tasks is one of the fundamental problems in computer vision. Data mixing techniques largely improve the quality of deep neural networks (DNNs) in various scenarios. Since mixup techniques are used as augmentations or auxiliary tasks in a wide range of cases, this repo focuses on mixup-related methods for Supervised, Self- and Semi-Supervised Representation Learning. Thus, we name this repo `OpenMixp`.

### Major features

This repo will be continued to update to support more self-supervised and mixup methods. Please watch us for latest update!

## Change Log

Please refer to [Change Log](docs/en/changelog.md) for details and release history.

[2020-07-07] `OpenMixup` v0.2.4 is released.

[2020-06-13] `OpenMixup` v0.2.3 is released.

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

Please refer to [Install](docs/en/install.md) for more detailed installation and dataset preparation.

## Get Started

Please see [Getting Started](docs/en/get_started.md) for the basic usage of OpenMixup (based on [MMSelfSup](https://github.com/open-mmlab/mmselfsup)).
Then, see [tutorials](docs/en/tutorials) for more tech details (based on MMClassification), which is similar to most open-source projects in MMLab.

## Benchmark and Model Zoo

[Model Zoos](docs/en/model_zoos) and lists of [Awesome Mixups](docs/en/awesome_mixups) have been released, and will be updated in the next two months. Checkpoints and traning logs will be updated soon!

<details close>
<summary>Currently supported backbone architectures</summary>

- [x] [VGG [ICLR'2015]](https://arxiv.org/abs/1409.1556)
- [x] [ResNet [CVPR'2016]](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- [x] [ResNeXt [CVPR'2017]](https://arxiv.org/abs/1611.05431)
- [x] [SE-ResNet [CVPR'2018]](https://arxiv.org/abs/1709.01507)
- [x] [SE-ResNeXt [CVPR'2018]](https://arxiv.org/abs/1709.01507)
- [x] [ShuffleNetV2 [ECCV'2018]](https://arxiv.org/abs/1807.11164)
- [x] [MobileNetV2 [CVPR'2018]](https://arxiv.org/abs/1801.04381)
- [x] [MobileNetV3 [ICCV'2019]](https://arxiv.org/abs/1905.02244)
- [x] [EfficientNet [ICML'2019]](https://arxiv.org/abs/1905.11946)
- [x] [Swin-Transformer [ICCV'2021]](https://arxiv.org/pdf/2103.14030.pdf)
- [x] [RepVGG [CVPR'2021]](https://arxiv.org/abs/2101.03697)
- [x] [Vision-Transformer [ICLR'2021]](https://arxiv.org/pdf/2010.11929.pdf)
- [x] [MLP-Mixer [NIPS'2021]](https://arxiv.org/abs/2105.01601)
- [x] [DeiT [ICML'2021]](https://arxiv.org/abs/2012.12877)
- [x] [ConvMixer [Openreview'2021]](https://arxiv.org/abs/2201.09792)
- [x] [PoolFormer [CVPR'2022]](https://arxiv.org/abs/2111.11418)
- [x] [ConvNeXt [CVPR'2022]](https://arxiv.org/abs/2201.03545)
- [x] [VAN [ArXiv'2022]](https://arxiv.org/abs/2202.09741)
</details>

<details close>
<summary>Currently supported mixup methods for supervised learning</summary>

- [x] [Mixup [ICLR'2018]](https://arxiv.org/abs/1710.09412)
- [x] [CutMix [ICCV'2019]](https://arxiv.org/abs/1905.04899)
- [x] [ManifoldMix [ICML'2019]](https://arxiv.org/abs/1806.05236)
- [x] [FMix [ArXiv'2020]](https://arxiv.org/abs/2002.12047)
- [x] [AttentiveMix [ICASSP'2020]](https://arxiv.org/abs/2003.13048)
- [x] [SmoothMix [CVPRW'2020]](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.pdf)
- [x] [SaliencyMix [ICLR'2021]](https://arxiv.org/abs/1710.09412)
- [x] [PuzzleMix [ICML'2020]](https://arxiv.org/abs/2009.06962)
- [x] [GridMix [Pattern Recognition'2021]](https://www.sciencedirect.com/science/article/pii/S0031320320303976)
- [x] [ResizeMix [ArXiv'2020]](https://arxiv.org/abs/2012.11101)
- [x] [AutoMix [ECCV'2022]](https://arxiv.org/abs/2103.13027)
- [x] [SAMix [ArXiv'2021]](https://arxiv.org/abs/2111.15454)
</details>

<details close>
<summary>Currently supported self-supervised algorithms</summary>

- [x] [Relative Location [ICCV'2015]](https://arxiv.org/abs/1505.05192)
- [x] [Rotation Prediction [ICLR'2018]](https://arxiv.org/abs/1803.07728)
- [x] [DeepCluster [ECCV'2018]](https://arxiv.org/abs/1807.05520)
- [x] [NPID [CVPR'2018]](https://arxiv.org/abs/1805.01978)
- [x] [ODC [CVPR'2020]](https://arxiv.org/abs/2006.10645)
- [x] [MoCov1 [CVPR'2020]](https://arxiv.org/abs/1911.05722)
- [x] [SimCLR [ICML'2020]](https://arxiv.org/abs/2002.05709)
- [x] [MoCov2 [ArXiv'2020]](https://arxiv.org/abs/2003.04297)
- [x] [BYOL [NIPS'2020]](https://arxiv.org/abs/2006.07733)
- [x] [SwAV [NIPS'2020]](https://arxiv.org/abs/2006.09882)
- [x] [DenseCL [CVPR'2021]](https://arxiv.org/abs/2011.09157)
- [x] [SimSiam [CVPR'2021]](https://arxiv.org/abs/2011.10566)
- [x] [Barlow Twins [ICML'2021]](https://arxiv.org/abs/2103.03230)
- [x] [MoCo v3 [ICCV'2021]](https://arxiv.org/abs/2104.02057)
- [x] [MAE [CVPR'2022]](https://arxiv.org/abs/2111.06377)
- [x] [SimMIM [CVPR'2022]](https://arxiv.org/abs/2111.09886)
- [x] [CAE [ArXiv'2022]](https://arxiv.org/abs/2202.03026)
- [x] [A2MIM [ArXiv'2022]](https://arxiv.org/abs/2205.13943)
</details>

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

- OpenMixup is an open-source project for mixup methods created by researchers in CAIRI AI LAB. We encourage researchers interested in visual representation learning and mixup methods to contribute to OpenMixup!
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

For now, the direct contributors include: Siyuan Li ([@Lupin1998](https://github.com/Lupin1998)), Zicheng Liu ([@pone7](https://github.com/pone7)), and Di Wu ([@wudi-bu](https://github.com/wudi-bu)). We thanks contributors for MMSelfSup and MMClassification.

## Contact

This repo is currently maintained by Siyuan Li (lisiyuan@westlake.edu.cn) and Zicheng Liu (liuzicheng@westlake.edu.cn).
