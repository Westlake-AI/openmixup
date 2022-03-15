
# OpenMixup

**News**
* OpenMixup v0.1.1 is now released, which supports various backbones, various mixup methods (e.g., [PuzzleMix](https://arxiv.org/abs/2009.06962), [AutoMix](https://arxiv.org/pdf/2103.13027), [SAMix](https://arxiv.org/pdf/2111.15454), etc.), various classification datasets, benchmarks in model_zoo.
* OpenMixup v0.1.0 is released, which supports various backbones (ConvNets and ViTs), config files generation, FP16 training (Apex or MMCV).

## Introduction

The master branch works with **PyTorch 1.6** or higher.

`OpenMixup` is an open-source supervised, self- and semi-unsupervised representation learning toolbox based on PyTorch, especially for mixup-related methods.

### What does this repo do?

Learning discriminative visual representation efficiently that facilitates downstream tasks is one of the fundamental problems in computer vision. Data mixing techniques largely improve the quality of deep neural networks (DNNs) in various scenarios. Since mixup techniques are used as augmentations or auxiliary tasks in a wide range of cases, this repo focuses on mixup-related methods for Supervised, Self- and Semi-Supervised Representation Learning. Thus, we name this repo `OpenMixp`.

### Major features

This repo will be continued to update in the next two months! Please watch us for latest update!

## Change Log

Please refer to [CHANGELOG.md](docs/CHANGELOG.md) for details and release history.

[2020-03-15] `OpenMixup` v0.1.1 is released.
[2020-01-22] `OpenMixup` v0.1.0 is released (bugs fixed).

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

## Get Started

Please see [Getting Started](docs/GETTING_STARTED.md) for the basic usage of OpenMixup (based on MMSelfSup).
Then, see [tutorials](docs/tutorials) for more tech details (based on MMClassification).

## Benchmark and Model Zoo

[Model Zoos](docs/model_zoos) and list of [Awesome Mixups](docs/awesome_mixups) will be updated in the next two months (before 2022.04)!

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

- OpenMixup is an open source project for mixup methods created by researchers in CAIRI AI LAB. We encourage researchers interested in mixup methods to contribute to OpenMixup!
- This repo borrows the architecture design and part of the code from [MMSelfSup](https://github.com/open-mmlab/mmselfsup) and [MMClassification](https://github.com/open-mmlab/mmclassification).

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2022openmixup,
    title={{OpenMixup}: Open Mixup Toolbox and Benchmark for Visual Representation},
    author={Li, Siyuan and Liu, Zichen and Wu, Di},
    howpublished = {\url{https://github.com/Westlake-AI/openmixup}},
    year={2022}
}
```

## Contributors

For now, the direct contributors include: Siyuan Li ([@Lupin1998](https://github.com/Lupin1998)), Zicheng Liu ([@pone7](https://github.com/pone7)), and Di Wu ([@wudi-bu](https://github.com/wudi-bu)). We thanks contributors for MMSelfSup and MMClassification.

## Contact

This repo is currently maintained by Siyuan Li (lisiyuan@westlake.edu.cn) and Zicheng Liu (liuzicheng@westlake.edu.cn).
