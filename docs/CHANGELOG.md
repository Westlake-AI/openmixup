## Changelog

### v0.2.0 (31/03/2022)

#### Highlight
* Support various popular backbones (ConvNets and ViTs), various image datasets, popular mixup methods, and benchmarks for supervised learning. Config files are available.
* Support popular self-supervised methods (e.g., BYOL, MoCo.V3, MAE) on both large-scale and small-scale datasets, and self-supervised benchmarks (merged from MMSelfSup). Config files are available.
* Support analyzing tools for self-supervised learning (kNN/SVM/linear metrics and t-SNE/UMAP visualization).
* Convenient usage of configs: fast configs generation by 'auto_train.py' and configs inheriting (MMCV).
* Support mixed-precision training (NVIDIA Apex or MMCV Apex).

#### Bug Fixes
* Done code refactoring follows MMSelfSup and MMClassification.

### v0.1.3 (25/03/2022)

* Refactor code structures for vision transformers and self-supervised methods (e.g., MoCo.V3 and MAE).
* Provide online analysis of self-supervised methods (knn metric and t-SNE/UMAP visualization). 
* More results are provided in Model Zoos.

#### Bug Fixes
* Fix bugs of reusing of configs, ViTs, visualization tools, etc. It requires rebuilding of OpenMixup (install mmcv-full).

### v0.1.2 (20/03/2022)

#### Highlight
* Refactor code structures according to MMSelfsup to fit high version of mmcv and PyTorch.
* Support self-supervised methods and optimizes config structures.

### v0.1.1 (15/03/2022)

#### Highlight
* Support various popular backbones (ConvNets and ViTs).
* Support various handcrafted methods and optimization-based methods (e.g., [PuzzleMix](https://arxiv.org/abs/2009.06962), [AutoMix](https://arxiv.org/pdf/2103.13027), [SAMix](https://arxiv.org/pdf/2111.15454), etc.).
* Provide supervised image classification benchmarks in model_zoo and results (on updating). 

#### Bug Fixes
* Fix bugs of new mixup methods (e.g., gco for Puzzlemix, etc.).

### v0.1.0 (22/01/2022)

#### Highlight
* Support various popular backbones (popular ConvNets and ViTs).
* Support mixed precision training (NVIDIA Apex or MMCV Apex).
* Support supervised, self- & semi-supervised learning methods and benchmarks.
* Support fast configs generation from a basic config file by 'auto_train.py'. 

#### Bug Fixes
* Fix bugs of code refactoring (backbones, fp16, etc.).

#### OpenSelfSup (v0.3.0, 14/10/2020) Supported Features

* Mixed Precision Training (NVIDIA Apex).
* Improvement of GaussianBlur doubles the training speed of MoCo V2, SimCLR, BYOL.
* More benchmarking results, including Places, VOC, COCO, linear/semi-supevised benchmarks.
* Fix bugs in moco v2 and byol, now the results are reproducible.
* Provide benchmarking results and model download links.
* Support updating network every several interations (accumulation).
* Support LARS and LAMB optimizer with nesterov (LAMB from MMclassification).
* Support excluding specific parameters from optimizer updation.
