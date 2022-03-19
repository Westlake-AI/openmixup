## Changelog

### v0.1.2 (20/03/2022)

#### Highlight
* Refactor code structures according to MMSelfsup to fit high version of mmcv and PyTorch.
* Support self-supervised methods and optimizes config structures.

### v0.1.1 (15/03/2022)

#### Highlight
* Support various popular backbones (ConvNets and ViTs).
* Support various handcrafted methods and optimization-based methods (e.g., PuzzleMix, AutoMix, SAMix).
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
