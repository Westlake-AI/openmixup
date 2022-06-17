
# OpenMixup

**News**
* OpenMixup v0.2.3 is released, which supports new self-supervised and mixup methods (e.g., [A2MIM](https://arxiv.org/abs/2205.13943)) and backbones ([UniFormer](https://arxiv.org/abs/2201.09450)), update the [online document](https://westlake-ai.github.io/openmixup/) and config files, and adds new features as [#6](https://github.com/Westlake-AI/openmixup/issues/6).
* OpenMixup v0.2.2 is released, which supports new self-supervised methods ([BarlowTwins](https://arxiv.org/abs/2103.03230), [SimMIM](https://arxiv.org/abs/2111.09886), etc.), backbones ([ConvMixer](https://arxiv.org/pdf/2201.09792.pdf), [MLPMixer](https://arxiv.org/pdf/2105.01601.pdf), [VAN](https://arxiv.org/pdf/2202.09741v2.pdf), etc.), and losses as [#5](https://github.com/Westlake-AI/openmixup/issues/5).
* OpenMixup v0.2.1 is released, which supports new methods as [#4](https://github.com/Westlake-AI/openmixup/issues/4) (bugs fixed).
* OpenMixup v0.2.0 is released, which supports new features as [#3](https://github.com/Westlake-AI/openmixup/issues/3). We have reorganized configs and fixed bugs.
* OpenMixup v0.1.3 is released (finished code refactoring and fixed bugs), which steadily supports ViTs, self-supervised methods (e.g., [MoCo.V3](https://arxiv.org/abs/2104.02057) and [MAE](https://arxiv.org/abs/2111.06377)), and online analysis (kNN metric and visualization). It requires the rebuilding of OpenMixup (install mmcv-full to support ViTs). More results are provided in Model Zoos.
* OpenMixup v0.1.1 is released, which supports various backbones (ConvNets and ViTs), various mixup methods (e.g., [PuzzleMix](https://arxiv.org/abs/2009.06962), [AutoMix](https://arxiv.org/pdf/2103.13027), [SAMix](https://arxiv.org/pdf/2111.15454), [DecoupleMix](https://arxiv.org/abs/2203.10761) etc.), various classification datasets, benchmarks (model_zoo), config files generation, FP16 training (Apex or MMCV).

## Introduction

The master branch works with **PyTorch 1.8** or higher (required by some self-supervised methods). You can still use **PyTorch 1.6** for supervised classification methods.

`OpenMixup` is an open-source toolbox for supervised, self-, and semi-supervised visual representation learning with mixup based on PyTorch, especially for mixup-related methods.

### What does this repo do?

Learning discriminative visual representation efficiently that facilitates downstream tasks is one of the fundamental problems in computer vision. Data mixing techniques largely improve the quality of deep neural networks (DNNs) in various scenarios. Since mixup techniques are used as augmentations or auxiliary tasks in a wide range of cases, this repo focuses on mixup-related methods for Supervised, Self- and Semi-Supervised Representation Learning. Thus, we name this repo `OpenMixp`.

### Major features

This repo will be continued to update to support more self-supervised and mixup methods. Please watch us for latest update!

## Change Log

Please refer to [Change Log](docs/en/changelog.md) for details and release history.

[2020-06-13] `OpenMixup` v0.2.3 is released.

[2020-05-24] `OpenMixup` v0.2.2 is released.

## Installation

Please refer to [Install](docs/en/install.md) for installation and dataset preparation.

## Get Started

Please see [Getting Started](docs/en/get_started.md) for the basic usage of OpenMixup (based on [MMSelfSup](https://github.com/open-mmlab/mmselfsup)).
Then, see [tutorials](docs/en/tutorials) for more tech details (based on MMClassification), which is similar to most open-source projects in MMLab.

## Benchmark and Model Zoo

[Model Zoos](docs/en/model_zoos) and lists of [Awesome Mixups](docs/en/awesome_mixups) have been released, and will be updated in the next two months. Checkpoints and traning logs will be updated soon! 

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
