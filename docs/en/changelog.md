## Changelog

### v0.2.10 (19/03/2025 till now)

Bump version to V0.2.10, supporting PyTorch=>2.0.0 and new mixup augmentations and backbones.

#### New Features

- Support new mixup augmentation methods. Config files and models & logs were provided and are on updating, including [TokenMix](https://arxiv.org/abs/2207.08409), [MixPro](https://arxiv.org/abs/2304.12043), [TLA](https://arxiv.org/abs/2210.06455), and [SUMix](https://arxiv.org/abs/2407.07805).

### Bug Fixes

- Fix some bugs with PyTorch=>2.0.0, including `AttributeError: 'MMDistributedDataParallel' object has no attribute "_use_replicated_tensor_module"` in `openmixup/api/train.py`, the parser error with `--local-rank` in `tools/train.py`.
- Remove `tools/single_train.sh` in openmixup. We should only start training with DDP using `dist_train.sh` variants.

### v0.2.9 (23/12/2023 till now)

Bump version to V0.2.9 with new mixup augmentations and various optimizers.

#### New Features

- Support new mixup augmentation methods, including [AdAutoMix](https://arxiv.org/abs/2312.11954) and [SnapMix](https://arxiv.org/abs/2012.04846). Config files and models & logs were provided and are on updating.
- Support more backbone architectures, including [UniRepLKNet](https://arxiv.org/abs/2311.15599), [TransNeXt](https://arxiv.org/abs/2311.17132), [StarNet](https://arxiv.org/abs/2403.19967), etc. Fixed some bugs.
- Support classical self-supervised method [DINO](https://arxiv.org/abs/2104.14294) with ViT-Base on ImageNet-1K.
- Support more PyTorch optimizers implemented, including Adam variants (e.g., AdaBelief, AdaFactor) and SGD variants (e.g., SGDP).
- Support evaluation tools for mixup augmentations, including robustness testing (corruption and adversiral attack robustness) and calibration evaluation.
- Provide more config files for self-supervised learning methods on small-scale datasets (CIFAR-100 and STL-10).
- Support [Sharpness-Aware Minimization (SAM)](https://openreview.net/forum?id=6Tm1mposlrM) optimizer variants for small-scale datasets.

### v0.2.8 (25/05/2023)

Bump version to V0.2.8 with new features in [MMPreTrain](https://github.com/open-mmlab/mmpretrain).

#### New Features

- Support more backbone architectures, including [MobileNetV3](https://arxiv.org/abs/1905.02244), [EfficientNetV2](https://arxiv.org/abs/2104.00298), [HRNet](https://arxiv.org/abs/1908.07919), [CSPNet](https://arxiv.org/abs/1911.11929), [LeViT](https://arxiv.org/abs/2104.01136), [MobileViT](http://arxiv.org/abs/2110.02178), [DaViT](https://arxiv.org/abs/2204.03645), and [MobileOne](http://arxiv.org/abs/2206.04040), etc.
- Support CIFAR-100 benchmarks of Metaformer architectures and Mixup variants with Transformers, detailed in [cifar100/advanced](https://github.com/Westlake-AI/openmixup/blob/main/configs/classification/cifar100/advanced) and [cifar100/mixups](https://github.com/Westlake-AI/openmixup/blob/main/configs/classification/cifar100/mixups). Models and logs of various CIFAR-100 mixup benchmarks are on updating.
- Support regression tasks with relavent datasets, metrics, and [configs](https://github.com/Westlake-AI/openmixup/blob/main/configs/regression). Datasets include [AgeDB](https://ieeexplore.ieee.org/document/8014984), [IMDB-WIKI](https://link.springer.com/article/10.1007/s11263-016-0940-3), and [RCFMNIST](https://arxiv.org/abs/2210.05775).
- Support Switch EMA in image classification, contrastive learning (BYOL, MoCo variants), and regression tasks.
- Support optimizers implemented in timm, including AdaBelief, AdaFactor, Lion, etc.

### Update Documents

- Update formats of awesome lists in [Awesome Mixups](docs/en/awesome_selfsup/MIM.md) and [Awesome MIM](docs/en/awesome_selfsup/MIM.md) and provide the latest methods (updated to 30/09/2023).

### Bug Fixes

- Fix the `by_epoch` setting in `CustomSchedulerHook` and update `DecoupleMix` in `soft_mix_cross_entropy` to support label smoothing settings.
- Fix bugs of Vision Transformers in [cls_mixup_head](https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/heads/cls_mixup_head.py) and [reg_head](https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/heads/reg_head.py).

### v0.2.7 (16/12/2022)

Bump version to V0.2.7 with new features as [#35](https://github.com/Westlake-AI/openmixup/issues/35). Update new features of `OpenMixup` v0.2.7 as issue [#36](https://github.com/Westlake-AI/openmixup/issues/36).

#### Code Refactoring

- Refactor `openmixup.core` (instead of `openmixup.hooks`) and `openmixup.models.augments` (contains mixup augmentation methods which are originally implemented in `openmixup.models.utils`). After code refactoring, the macro design of `OpenMixup` is similar to most projects of MMLab.
- Support deployment of `ONNX` and `TorchScript` in `openmixup.core.export` and `tools/deployment`. We refactored the abstract class `BaseModel` (implemented in `openmixup/models/classifiers/base_model.py`) to support `forward_inference` (for custom inference and visualization). We also refactored `openmixup.models.heads` and `openmixup.models.losses` to support `forward_inference`. You can deploy the classification models in `OpenMixup` according to [deployment tutorials](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/tools).
- Support testing API methods in `openmixup/apis/test.py` for evaluation and deployment of classification models.
- Refactor `openmixup.core.optimizers` to separate optimizers and builders and support the latest [Adan](https://arxiv.org/abs/2208.06677) optimizer.
- Refactor [`mixup_classification.py`](https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/classifiers/mixup_classification.py) to support label mixup methods, add `return_mask` for mixup methods in [`augments`](https://github.com/Westlake-AI/openmixup/tree/main/openmixup/models/augments) and add `return_attn` in ViT backbone.
- Refactor `ValidateHook` to support new features as `EvalHook` in mmcv, e.g., `save_best="auto"` during training.
- Refactor `ClsHead` with `BaseClsHead` to support MLP classification head variants in modern network architectures.

#### New Features

- Support detailed usage instructions in README of config files for image classification methods in `configs/classification`, e.g., [mixups on ImageNet](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/README.md). READMEs of other methods in `configs/selfsup` and `configs/semisup` will also be updated.
- Refine the origianzation of README files according to [README-Template](https://github.com/othneildrew/Best-README-Template).
- Support the new mixup augmentation method ([AlignMix](https://arxiv.org/abs/2103.15375)) and provide the relevant config files in various datasets.
- Refine the setup for the local installation and PyPi release in `setup.py` and `setup.cfg`. View PyPi project of [OpenMixup](https://pypi.org/project/openmixup).
- Support a new mixup method [TransMix](https://arxiv.org/abs/2111.09833) and provide config files in [mixups/deit](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/deit).
- Update config files. Provide full config files of mixup methods based on ViT-T/S/B on ImageNet and update [RSB A3](https://arxiv.org/abs/2110.00476) config files for popular backbones.
- Update `target_generators` to support the latest MIM pre-training methods (fixed requirements).
- Update config files and scripts for SSL downstream tasks benchmarks (classification, detection, and segmentation).
- Update and fix bugs in visualization tools ([vis_loss_landscape](https://github.com/Westlake-AI/openmixup/tree/main/tools/visualizations/vis_loss_landscape.py)). Fix [model converters](https://github.com/Westlake-AI/openmixup/tree/main/tools/model_converters) tools.
- Support [Semantic-Softmax](https://arxiv.org/abs/2104.10972) loss and [ImageNet-21K-P (Winter)](https://openreview.net/forum?id=Zkj_VcZ6ol&noteId=1oUacUMpIbg) pre-training.
- Support more backbone architectures, including [BEiT](https://arxiv.org/abs/2106.08254), [MetaFormer](https://arxiv.org/abs/2210.13452), [ConvNeXtV2](http://arxiv.org/abs/2301.00808), [VanillaNet](https://arxiv.org/abs/2305.12972), and [CoC](https://arxiv.org/abs/2303.01494).

### Update Documents

- Update documents of mixup benchmarks on ImageNet in [Model_Zoo_sup.md](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md). Update config files for supported mixup methods.
- Update formats (figures, introductions and content tables) of awesome lists in [Awesome Mixups](docs/en/awesome_selfsup/MIM.md) and [Awesome MIM](docs/en/awesome_selfsup/MIM.md) and provide the latest methods (updated to 18/03/2023).
- Update `api` that describes the overall code structures in `docs/en/api` for the readthedocs page.
- Reorganize and update tutorials for SSL downstream tasks benchmarks (classification, detection, and segmentation).

### v0.2.6 (41/09/2022)

Bump version to V0.2.6 with new features as [#20](https://github.com/Westlake-AI/openmixup/issues/20). Update new features and documents of `OpenMixup` v0.2.6 as issue [#24](https://github.com/Westlake-AI/openmixup/issues/24), fix relevant issue [#25](https://github.com/Westlake-AI/openmixup/issues/25), issue [#26](https://github.com/Westlake-AI/openmixup/issues/26), issue [#27](https://github.com/Westlake-AI/openmixup/issues/27), issue [#31](https://github.com/Westlake-AI/openmixup/issues/31), and issue [#33](https://github.com/Westlake-AI/openmixup/issues/33).

#### New Features

- Support new backbone architectures ([EdgeNeXt](https://arxiv.org/abs/2206.10589), [EfficientFormer](https://arxiv.org/abs/2206.01191), [HorNet](https://arxiv.org/abs/2207.14284), ([MogaNet](https://arxiv.org/abs/2211.03295), [MViT.V2](https://arxiv.org/abs/2112.01526), [ShuffleNet.V1](https://arxiv.org/abs/1707.01083), [DeiT-3](https://arxiv.org/abs/2204.07118)), and provide relevant network modules in `models/utils/layers`. Config files and README.md are updated.
- Support new self-supervised method [BEiT](https://arxiv.org/abs/2106.08254) with ViT-Base on ImageNet-1K, and fix bugs of [CAE](https://arxiv.org/abs/2202.03026), [MaskFeat](https://arxiv.org/abs/2112.09133), and [SimMIM](https://arxiv.org/abs/2111.09886) in `Dataset`, `Model`, and `Head`. Note that we added `HOG` feature implementation borrowed from the [original repo](https://github.com/facebookresearch/SlowFast) for [MaskFeat](https://arxiv.org/abs/2112.09133). Update pre-training and fine-tuning config files, and documents for the relevant masked image modeling (MIM) methods ([BEiT](https://arxiv.org/abs/2106.08254), [MaskFeat](https://arxiv.org/abs/2111.06377), [CAE](https://arxiv.org/abs/2202.03026), and [A2MIM](https://arxiv.org/abs/2205.13943)). Support more fine-tuning setting on ImageNet for MIM pre-training based on various backbones (e.g., ViTs, ResNets, ConvNeXts).
- Fix the updated arXiv.V2 version of [VAN](https://arxiv.org/pdf/2202.09741v2.pdf) by adding architecture configurations.
- Support [ArcFace](https://arxiv.org/abs/1801.07698) loss for metric learning and the relevant `NormLinearClsHead`. And support [SeeSaw](https://arxiv.org/abs/2008.10032) loss for long-tail classification tasks.
- Update the issue template with more relevant links and emojis.
- Support Grad-CAM visualization tools [vis_cam.py](tools/visualizations/vis_cam.py) of supported architectures.

### Update Documents

- Update our `OpenMixup` tech report on [arXiv](https://arxiv.org/abs/2209.04851), which provides more technical details and benchmark results.
- Update self-supervised learning [Model_Zoo_selfsup.md](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_selfsup.md). And update documents of the new backbone and self-supervised methods.
- Update supervised learning [Model_Zoo_sup.md](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md) as provided in [AutoMix](https://arxiv.org/abs/2103.13027) and support more mixup benchmark results.
- Update the template and add the latest paper lists of mixup and MIM methods in [Awesome Mixups](docs/en/awesome_selfsup/MIM.md) and [Awesome MIM](docs/en/awesome_selfsup/MIM.md). We provide teaser figures of most papers as illustrations.
- Update [documents](docs/en/tools) of `tools`.

### Bug Fixes

- Fix raising error notification of `torch.fft` for *PyTorch 1.6* or lower versions in backbones and heads.
- Fix `README.md` (new icons, fixing typos) and support pytest in `tests`.
- Fix the classification heads and update implementations and config files of [AlexNet](https://dl.acm.org/doi/10.1145/3065386) and [InceptionV3](https://arxiv.org/abs/1512.00567).

### v0.2.5 (21/07/2022)

Bump version to V0.2.5 with new features and updating documents as [#10](https://github.com/Westlake-AI/openmixup/issues/10). Update features and fix bugs in V0.2.5 as [#17](https://github.com/Westlake-AI/openmixup/issues/17). Update features and documents in V0.2.5 as [#18](https://github.com/Westlake-AI/openmixup/issues/18) and [#19](https://github.com/Westlake-AI/openmixup/issues/19).

#### New Features

- Support new attention mechanisms in backbone architectures ([Anti-Oversmoothing](https://arxiv.org/abs/2203.05962), `FlowAttention` in [FlowFormer](https://arxiv.org/abs/2202.06258) and `PoolAttention` in [MViTv2](https://arxiv.org/abs/2112.01526)).
- Update code intergration testing in [tests](https://github.com/Westlake-AI/openmixup/tests/).

### Update Documents

- Recognize `README` and `README` for various methods.
- Update [Awesome Mixups](docs/en/awesome_selfsup/MIM.md) and [Awesome MIM](docs/en/awesome_selfsup/MIM.md).
- Update [get_started.md](docs/en/get_started.md) and [Tutorials](docs/en/tutorials) for better usage of `OpenMixup`.
- Update mixup benchmarks in [model_zoos](docs/en/model_zoos/Model_Zoo_sup.md): providing configs, weights, and more details.
- Update latest methods in [Awesome Mixups](docs/en/awesome_selfsup/MIM.md) and [Awesome MIM](docs/en/awesome_selfsup/MIM.md).
- Update `README.md` and fix `auto_train_mixups.py` for various datasets.

### Bug Fixes

- Fix visualization of the reconstruction results in `MAE`.
- Fix the normalization bug in config files and `plot_torch.py` as mentioned in [#16](https://github.com/Westlake-AI/openmixup/issues/16).
- Fix the random seeds in `tools/train.py` as mentioned in [#14](https://github.com/Westlake-AI/openmixup/issues/14).

### v0.2.4 (07/07/2022)

Update new features and fix bugs as [#7](https://github.com/Westlake-AI/openmixup/issues/7).

#### New Features

- Support new backbone architectures ([LITv2](https://arxiv.org/abs/2205.13213)).
- Refactor code structures weight initialization in various network modules (using `BaseModule` in `mmcv`).
- Refactor code structures of `openmixup.models.utils.layers` to support more network structures.

### Bug Fixes

- Fix bugs that cause degenerate performances of pure Transformer backbones (DeiT and Swin) in `OpenMixup`. The main reason might be the old version of `auto_fp16` and `DistOptimizerHook` implementations, since `PyTorch=>1.6.0` has better support of fp16 training than `mmcv`.
- Fix the bug of ViT fine-tuning for MIM methods (e.g., MAE, SimMIM). The original `MIMVisionTransformer` in `openmixup.models.mim_vit` has frozen all the backbone parameters during fine-tuning.
- Fix the initialization of Transformer-based architectures (e.g., ViT, Swin) to reproduce the train-from-scratch performances.
- Fix the weight initialization of Transformer-based architectures (e.g., ViT, Swin) to reproduce the train-from-scratch performance. Update weight initialization, parameter-wise weight decay, and fp16 settings in relevant config files.

### v0.2.3 (17/06/2022)

Support new features as [#6](https://github.com/Westlake-AI/openmixup/issues/6).

#### New Features

- Support the [online document](https://westlake-ai.github.io/openmixup/) of OpenMixup (built on Read the Docs).
- Provide README and update configs for [self-supervised](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/) and [supervised](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/) methods.
- Support new Masked Image Modeling (MIM) methods ([A2MIM](https://arxiv.org/abs/2205.13943), [CAE](https://arxiv.org/abs/2202.03026)).
- Support new backbone networks ([DenseNet](https://arxiv.org/abs/1608.06993), [ResNeSt](https://arxiv.org/abs/2004.08955), [PoolFormer](https://arxiv.org/abs/2111.11418), [UniFormer](https://arxiv.org/abs/2201.09450)).
- Support new Fine-tuing method ([HCR](https://arxiv.org/abs/2206.00845)).
- Support new mixup augmentation methods ([SmoothMix](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.pdf), [GridMix](https://www.sciencedirect.com/science/article/pii/S0031320320303976)).
- Support more regression losses ([Focal L1/L2 loss](https://arxiv.org/abs/2102.09554), [Balanced L1 loss](https://arxiv.org/abs/1904.02701), [Balanced MSE loss](https://arxiv.org/abs/2203.16427)).
- Support more regression metrics (regression errors and correlations) and the regression dataset.
- Support more reweight classification losses ([Gradient Harmonized loss](https://arxiv.org/abs/1811.05181), [Varifocal Focal Loss](https://arxiv.org/abs/1811.05181)) from [MMDetection](https://github.com/open-mmlab/mmdetection).

### Bug Fixes

- Refactor code structures of `openmixup.models.utils` and support more network layers.
- Fix the bug of `DropPath` (using stochastic depth rule) in `ResNet` for RSB A1/A2 training settings.

### v0.2.2 (24/05/2022)

Support new features and finish code refactoring as [#5](https://github.com/Westlake-AI/openmixup/issues/5).

#### Highlight

- Support more self-supervised methods ([Barlow Twins](https://arxiv.org/abs/2103.03230) and Masked Image Modeling methods).
- Support popular backbones ([ConvMixer](https://arxiv.org/abs/2201.09792), [MLPMixer](https://arxiv.org/abs/2105.01601), [VAN](https://arxiv.org/abs/2202.09741)) based on MMClassification.
- Support more regression losses ([Charbonnier loss](https://arxiv.org/abs/1710.01992v1) and [Focal Frequency loss](https://arxiv.org/pdf/2012.12821.pdf)).

### Bug Fixes

- Fix bugs in self-supervised classification benchmarks (configs and implementations of VisionTransformer).
- Update [INSTALL.md](INSTALL.md). We suggest you install **PyTorch 1.8** or higher and mmcv-full for better usage of this repo. **PyTorch 1.8** has bugs in AdamW optimizer (do not use **PyTorch 1.8** to fine-tune ViT-based methods).
- Fix bugs in PreciseBNHook (update all BN stats) and RepeatSampler (set sync_random_seed).

### v0.2.1 (19/04/2022)

Support new features and finish code refactoring as [#4](https://github.com/Westlake-AI/openmixup/issues/4).

#### New Features

- Support masked image modeling (MIM) self-supervised methods ([MAE](https://arxiv.org/abs/2111.06377), [SimMIM](https://arxiv.org/abs/2111.09886), [MaskFeat](https://arxiv.org/abs/2112.09133)).
- Support visualization of reconstruction results in MIM methods.
- Support basic regression losses and metrics.

### Bug Fixes

- Fix bugs in regression metrics, MIM dataset, and benchmark configs. Notice that only `l1_loss` is supported by FP16 training, other regression losses (e.g., MSE and Smooth_L1 losses) will cause NAN when the target and prediction are not normalized in FP16 training.
- We suggest you install **PyTorch 1.8** or higher (required by some self-supervised methods) and `mmcv-full` for better usage of this repo. Do not use **PyTorch 1.8** to fine-tune ViT-based methods, and you can still use **PyTorch 1.6** for supervised classification methods.

### v0.2.0 (31/03/2022)

Support new features and finish code refactoring as [#3](https://github.com/Westlake-AI/openmixup/issues/3).

#### New Features

- Support various popular backbones (ConvNets and ViTs), various image datasets, popular mixup methods, and benchmarks for supervised learning. Config files are available.
- Support popular self-supervised methods (e.g., BYOL, MoCo.V3, MAE) on both large-scale and small-scale datasets, and self-supervised benchmarks (merged from MMSelfSup). Config files are available.
- Support analyzing tools for self-supervised learning (kNN/SVM/linear metrics and t-SNE/UMAP visualization).
- Convenient usage of configs: fast configs generation by 'auto_train.py' and configs inheriting (MMCV).
- Support mixed-precision training (NVIDIA Apex or MMCV Apex) for all methods.
- [Model Zoos](docs/model_zoos) and lists of [Awesome Mixups](docs/awesome_mixups) have been released.

#### Bug Fixes

- Done code refactoring follows MMSelfSup and MMClassification.

### v0.1.3 (25/03/2022)

- Refactor code structures for vision transformers and self-supervised methods (e.g., [MoCo.V3](https://arxiv.org/abs/2104.02057) and [MAE](https://arxiv.org/abs/2111.06377)).
- Provide online analysis of self-supervised methods (knn metric and t-SNE/UMAP visualization).
- More results are provided in Model Zoos.

#### Bug Fixes

- Fix bugs of reusing of configs, ViTs, visualization tools, etc. It requires rebuilding of OpenMixup (install mmcv-full).

### v0.1.2 (20/03/2022)

#### New Features

- Refactor code structures according to MMSelfsup to fit high version of mmcv and PyTorch.
- Support self-supervised methods and optimizes config structures.

### v0.1.1 (15/03/2022)

#### New Features

- Support various popular backbones (ConvNets and ViTs) and update config files.
- Support various handcrafted methods and optimization-based methods (e.g., [PuzzleMix](https://arxiv.org/abs/2009.06962), [AutoMix](https://arxiv.org/pdf/2103.13027), [SAMix](https://arxiv.org/pdf/2111.15454), [DecoupleMix](https://arxiv.org/abs/2203.10761), etc.). Config files generation of mixup methods are supported.
- Provide supervised image classification benchmarks in model_zoo and results (on updating).

#### Bug Fixes

- Fix bugs of new mixup methods (e.g., gco for Puzzlemix, etc.).

### v0.1.0 (22/01/2022)

#### New Features

- Support various popular backbones (popular ConvNets and ViTs).
- Support mixed precision training (NVIDIA Apex or MMCV Apex).
- Support supervised, self- & semi-supervised learning methods and benchmarks.
- Support fast configs generation from a basic config file by `auto_train.py`.

#### Bug Fixes

- Fix bugs of code refactoring (backbones, fp16 training, etc.).

#### OpenSelfSup (v0.3.0, 14/10/2020) Supported Features

This repo is originally built on OpenSelfSup (the old version of [MMSelfSup](https://github.com/open-mmlab/mmselfsup)) and borrows some implementations from [MMClassification](https://github.com/open-mmlab/mmclassification).

- Mixed Precision Training (based on NVIDIA Apex for **PyTorch 1.6**).
- Improvement of GaussianBlur doubles the training speed of MoCo V2, SimCLR, and BYOL.
- More benchmarking results, including benchmarks on Places, VOC, COCO, and linear/semi-supervised benchmarks.
- Fix bugs in moco v2 and BYOL so that the reported results are reproducible.
- Provide benchmarking results and model download links.
- Support updating the network every several iterations (accumulation).
- Support LARS and LAMB optimizer with Nesterov (LAMB from [MMClassification](https://github.com/open-mmlab/mmclassification)).
- Support excluding specific parameter-wise settings from the optimizer updating.
