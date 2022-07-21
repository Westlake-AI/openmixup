# Model Zoo of Supervised Learning

**OpenMixup provides mixup benchmarks on supervised learning on various tasks. Configs, experiment results, and training logs will be updated as soon as possible. More mixup variants will be supported!**

Now, we have supported 13 popular mixup methods! Notice that * denotes open-source arXiv pre-prints reproduced by us, and 📖 denotes original results reproduced by official implementations. We modified the original AttentiveMix by using pre-trained R-18 and sampling $\lambda$ from $\Beta(\alpha,8)$ as AttentiveMix+. Moreover, you can summarize experiment results (json files) by tools in `openmixup/tools/summary/`.

<details open>
<summary>Supported sample mixing policies</summary>

- [x] [Mixup [ICLR'2018]](https://arxiv.org/abs/1710.09412)
- [x] [CutMix [ICCV'2019]](https://arxiv.org/abs/1905.04899)
- [x] [ManifoldMix [ICML'2019]](https://arxiv.org/abs/1806.05236)
- [x] [FMix [ArXiv'2020]](https://arxiv.org/abs/2002.12047)
- [x] [AttentiveMix [ICASSP'2020]](https://arxiv.org/abs/2003.13048)
- [x] [SmoothMix [CVPRW'2020]](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.pdf)
- [x] [SaliencyMix [ICLR'2021]](https://arxiv.org/abs/1710.09412)
- [x] [PuzzleMix [ICML'2020]](https://arxiv.org/abs/2009.06962)
- [ ] [Co-Mixup [ICLR'2021]](https://openreview.net/forum?id=gvxJzw8kW4b)
- [x] [GridMix [Pattern Recognition'2021]](https://www.sciencedirect.com/science/article/pii/S0031320320303976)
- [ ] [SuperMix [CVPR'2021]](https://arxiv.org/abs/2003.05034)
- [x] [ResizeMix [ArXiv'2020]](https://arxiv.org/abs/2012.11101)
- [ ] [AlignMix [CVPR'2022]](https://arxiv.org/abs/2103.15375)
- [x] [AutoMix [ECCV'2022]](https://arxiv.org/abs/2103.13027)
- [x] [SAMix [ArXiv'2021]](https://arxiv.org/abs/2111.15454)
- [ ] [RecursiveMix [ArXiv'2022]](https://arxiv.org/abs/2203.06844)

</details>

<details open>
<summary>Supported label mixing policies</summary>

- [ ] [Saliency Grafting [AAAI'2022]](https://arxiv.org/abs/2112.08796v1)
- [ ] [TransMix [CVPR'2022]](https://arxiv.org/abs/2111.09833)
- [x] [DecoupleMix [ArXiv'2022]](https://arxiv.org/abs/2203.10761)
- [ ] [TokenMix [ECCV'2022]](https://arxiv.org/abs/2207.08409)

</details>

## ImageNet Benchmarks

We provide three popular benchmarks on ImageNet-1k based on various backbones. We also provide results on TinyImageNet-200 for fast training. The **median** of top-1 accuracy in the last 5/10 training epochs for 100/300 epochs is reported for ResNet variants, and the **best** top-1 accuracy is reported for DeiT training settings.

### PyTorch-style Training Settings on ImageNet-1k

**Note**

* These benchmarks follow PyTorch-style settings, training 100 and 300 epochs on ImageNet-1k.
* Please run configs in `configs/classification/imagenet/mixups/basic`, and modify epochs and mix_mode in `auto_train_in_mixups.py` to generate proper configs by yourself.
* Notice that 📖 denotes original results reproduced by official implementations.

| Backbones       |  ResNet-18 |  ResNet-34 |  ResNet-50 | ResNet-101 | ResNeXt-101 |
|-----------------|:----------:|:----------:|:----------:|:----------:|:-----------:|
| Epochs          | 100 epochs | 100 epochs | 100 epochs | 100 epochs |  100 epochs |
| Vanilla         |    70.04   |    73.85   |    76.83   |    78.18   |    78.71    |
| MixUp           |    69.98   |    73.97   |    77.12   |    78.97   |    79.98    |
| CutMix          |    68.95   |    73.58   |    77.17   |    78.96   |    80.42    |
| ManifoldMix     |    69.98   |    73.98   |    77.01   |    79.02   |    79.93    |
| SaliencyMix     |    69.16   |    73.56   |    77.14   |    79.32   |    80.27    |
| AttentiveMix+   |    68.57   |      -     |    77.28   |      -     |      -      |
| FMix*           |    69.96   |    74.08   |    77.19   |    79.09   |    80.06    |
| PuzzleMix       |    70.12   |    74.26   |    77.54   |    79.43   |    80.53    |
| Co-Mixup📖     |      -     |      -     |    77.60   |      -     |      -      |
| SuperMix📖     |      -     |      -     |    77.63   |      -     |      -      |
| ResizeMix*      |    69.50   |    73.88   |    77.42   |    79.27   |    80.55    |
| AlignMix📖     |      -     |      -     |    78.00   |      -     |      -      |
| Grafting📖     |      -     |      -     |    77.74   |      -     |      -      |
| AutoMix*        |    70.50   |    74.52   |    77.91   |    79.87   |    80.89    |
| SAMix*          |    70.83   |    74.95   |    78.06   |    80.05   |    80.98    |

| Backbones       |  ResNet-18 |  ResNet-34 |  ResNet-50 | ResNet-101 |
|-----------------|:----------:|:----------:|:----------:|:----------:|
| Epochs          | 300 epochs | 300 epochs | 300 epochs | 300 epochs |
| Vanilla         |    71.83   |    75.29   |    77.35   |    78.91   |
| MixUp           |    71.72   |    75.73   |    78.44   |    80.60   |
| CutMix          |    71.01   |    75.16   |    78.69   |    80.59   |
| ManifoldMix     |    71.73   |    75.44   |    78.21   |    80.64   |
| SaliencyMix     |    70.21   |    75.01   |    78.46   |    80.45   |
| FMix*           |    70.30   |    75.12   |    78.51   |    80.20   |
| PuzzleMix       |    71.64   |    75.84   |    78.86   |    80.67   |
| ResizeMix*      |    71.32   |    75.64   |    78.91   |    80.52   |
| AlignMix📖     |      -     |      -     |    79.32   |      -     |
| AutoMix*        |    72.05   |    76.10   |    79.25   |    80.98   |
| SAMix*          |    72.27   |    76.28   |    79.39   |    81.10   |

### Timm RSB A2/A3 Training Settings on ImageNet-1k

**Note**

* This benchmark follows timm RSB A2/A3 settings, training 300/100 epochs with the BCE loss on ImageNet-1k. RSB A3 is a fast
* Please run configs in `configs/classification/imagenet/mixups/rsb_a2` and `configs/classification/imagenet/mixups/rsb_a3`.

| Backbones   | ResNet-50 | ResNet-50 | Eff-B0 | Eff-B0 | Mob.V2 1x | Mob.V2 1x |
|-------------|:---------:|:---------:|:------:|:------:|:---------:|:---------:|
| Settings    |     A2    |     A3    |   A2   |   A3   |     A2    |     A3    |
| RSB         |   79.80   |   78.08   |  77.26 |  74.02 |   72.87   |   69.86   |
| MixUp       |           |   77.66   |  77.19 |  73.87 |   72.78   |   70.17   |
| CutMix      |   79.38   |   77.62   |  77.24 |  73.46 |   72.23   |   69.62   |
| ManifoldMix |   79.47   |   77.78   |  77.22 |  73.83 |   72.34   |   70.05   |
| SaliencyMix |   79.42   |   77.93   |  77.67 |  73.42 |   72.07   |   69.69   |
| FMix*       |   79.05   |   77.76   |  77.33 |  73.71 |   72.79   |   70.10   |
| PuzzleMix   |   79.78   |   78.02   |  77.35 |  74.10 |   72.85   |   70.04   |
| ResizeMix*  |   79.74   |   77.85   |  77.27 |  73.67 |   72.50   |   69.94   |
| AutoMix*    |           |   78.44   |  77.58 |  74.61 |   73.19   |   71.16   |
| SAMix       |           |   78.64   |  77.69 |  75.28 |   73.42   |   71.24   |

### DeiT Training Settings on ImageNet-1k

**Note**

* Since recently proposed transformer-based archetectures adopt mixups as parts of enssential augmentations, we report the mean of the best performance in trivals as their original paper. Notice that the performances of transformer-based archetectures are more difficult to reproduce than ResNet variants.
* Please run configs in `configs/classification/imagenet/deit/`, `configs/classification/imagenet/swin/`, and `configs/classification/imagenet/convnext/`.
* Notice that 📖 denotes original results reproduced by official implementations.

| Methods         | DeiT-Small | Swin-Tiny | ConvNeXt-Tiny |
|-----------------|:----------:|:---------:|:-------------:|
| Vanilla         |    73.57   |   78.95   |     79.22     |
| DeiT            |    79.80   |   81.20   |     82.10     |
| MixUp           |    77.72   |   80.23   |     80.88     |
| CutMix          |    79.54   |   80.59   |     81.57     |
| ManifoldMix     |      -     |     -     |     80.57     |
| AttentiveMix+   |    77.63   |           |     81.14     |
| SaliencyMix     |    78.70   |           |     81.33     |
| PuzzleMix       |    79.73   |           |     81.48     |
| FMix            |    77.37   |           |     81.04     |
| ResizeMix       |    76.79   |   80.73   |     81.64     |
| TransMix📖     |    80.70   |   81.80   |       -       |
| AutoMix         |    80.78   |           |     82.28     |
| SAMix           |    80.94   |           |     82.35     |

### TinyImageNet-200

**Note**

* This benchmark largely based on CIFAR settings, training 400 epochs on TinyImageNet-200.
* Please run configs in `configs/classification/tiny_imagenet/mixups/`.
* Notice that 📖 denotes original results reproduced by official implementations.

| Backbones          | ResNet-18 | ResNeXt-50 |
|--------------------|:---------:|:----------:|
| Vanilla            |   61.68   |    65.04   |
| MixUp              |   63.86   |    66.36   |
| CutMix             |   65.53   |    66.47   |
| ManifoldMix        |   64.15   |    67.30   |
| SaliencyMix        |   64.60   |    66.55   |
| AttentiveMix+      |   64.85   |    67.42   |
| FMix*              |   63.47   |    65.08   |
| GridMix📖         |     -     |    69.12   |
| PuzzleMix          |   65.81   |    67.83   |
| Co-Mixup📖        |   65.92   |    68.02   |
| ResizeMix*         |   63.74   |    65.87   |
| Grafting📖        |   64.84   |      -     |
| AlignMix📖        |   66.87   |      -     |
| AutoMix*           |   67.33   |    70.72   |
| SAMix*             |   68.89   |    72.18   |


## CIFAR-10/100 Benchmarks

CIFAR benchmarks based on ResNet variants. We report the median of top-1 accuracy in the last 10 training epochs.

### CIFAR-10

**Note**

* This benchmark follows CutMix settings, training 200/400/800/1200 epochs on CIFAR-10.
* Please run configs in `configs/classification/cifar10/mixups/`.
* Notice that 📖 denotes original results reproduced by official implementations.

| Backbones       |  ResNet-18 |  ResNet-18 |  ResNet-18 |  ResNet-18  |
|-----------------|:----------:|:----------:|:----------:|:-----------:|
| Epochs          | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
| Vanilla         |    94.87   |    95.10   |    95.50   |    95.59    |
| MixUp           |    95.70   |    96.55   |    96.62   |    96.84    |
| CutMix          |    96.11   |    96.13   |    96.68   |    96.56    |
| ManifoldMix     |    96.04   |    96.57   |    96.71   |    97.02    |
| SaliencyMix     |    96.05   |    96.42   |    96.20   |    96.18    |
| AttentiveMix+   |    96.21   |    96.45   |    96.63   |    96.49    |
| FMix*           |    96.17   |    96.53   |    96.18   |    96.01    |
| PuzzleMix       |    96.42   |    96.87   |    97.10   |    97.13    |
| ResizeMix*      |    96.16   |    96.91   |    96.76   |    97.04    |
| AlignMix📖     |            |            |            |    97.05    |
| AutoMix*        |    96.59   |    97.08   |    97.34   |    97.30    |
| SAMix*          |    96.67   |    97.16   |    97.50   |    97.41    |

| Backbones       | ResNeXt-50 | ResNeXt-50 | ResNeXt-50 |  ResNeXt-50 |
|-----------------|:----------:|:----------:|:----------:|:-----------:|
| Epochs          | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
| Vanilla         |    95.92   |    95.81   |    96.23   |    96.26    |
| MixUp           |    96.88   |    97.19   |    97.30   |    97.33    |
| CutMix          |    96.78   |    96.54   |    96.60   |    96.35    |
| ManifoldMix     |    96.97   |    97.39   |    97.33   |    97.36    |
| SaliencyMix     |    96.65   |    96.89   |    96.70   |    96.60    |
| AttentiveMix+   |    96.84   |    96.91   |    96.87   |    96.62    |
| FMix*           |    96.72   |    96.76   |    96.76   |    96.10    |
| PuzzleMix       |    97.05   |    97.24   |    97.37   |    97.34    |
| ResizeMix*      |    97.02   |    97.38   |    97.21   |    97.36    |
| AlignMix📖     |            |            |            |             |
| AutoMix*        |    97.19   |    97.42   |    97.65   |    97.51    |
| SAMix*          |    97.23   |    97.51   |    97.93   |    97.74    |

### CIFAR-100

**Note**

* This benchmark follows CutMix settings, training 200/400/800/1200 epochs on CIFAR-100. Please use wd=5e-4 for cutting-based methods (CutMix, AttributeMix+, SaliencyMix, FMix, ResizeMix) based on ResNeXt-50 for better performances.
* Please run configs in `configs/classification/cifar100/mixups/`.
* Notice that 📖 denotes original results reproduced by official implementations.

| Backbones       |  ResNet-18 |  ResNet-18 |  ResNet-18 |  ResNet-18  |
|-----------------|:----------:|:----------:|:----------:|:-----------:|
| Epoch           | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
| Vanilla         |    76.42   |    77.73   |    78.04   |    78.55    |
| MixUp           |    78.52   |    79.34   |    79.12   |    79.24    |
| CutMix          |    79.45   |    79.58   |    78.17   |    78.29    |
| ManifoldMix     |    79.18   |    80.18   |    80.35   |    80.21    |
| SaliencyMix     |    79.75   |    79.64   |    79.12   |    77.66    |
| AttentiveMix+   |    79.62   |    80.14   |    78.91   |    78.41    |
| FMix*           |    78.91   |    79.91   |    79.69   |    79.50    |
| PuzzleMix       |    79.96   |    80.82   |    81.13   |    81.10    |
| Co-Mixup📖     |    80.01   |    80.87   |    81.17   |    81.18    |
| ResizeMix*      |    79.56   |    79.19   |    80.01   |    79.23    |
| AlignMix📖     |            |            |            |    81.71    |
| AutoMix*        |    80.12   |    81.78   |    82.04   |    81.95    |
| SAMix*          |    81.21   |    81.97   |    82.30   |    82.41    |

| Backbones       | ResNeXt-50 | ResNeXt-50 | ResNeXt-50 |  ResNeXt-50 |  WRN-28-8  |
|-----------------|:----------:|:----------:|:----------:|:-----------:|:----------:|
| Epoch           | 200 epochs | 400 epochs | 800 epochs | 1200 epochs | 400 epochs |
| Vanilla         |    79.37   |    80.24   |    81.09   |    81.32    |    81.63   |
| MixUp           |    81.18   |    82.54   |    82.10   |    81.77    |    82.82   |
| CutMix          |    81.52   |    78.52   |    78.32   |    77.17    |    84.45   |
| ManifoldMix     |    81.59   |    82.56   |    82.88   |    83.28    |    83.24   |
| SaliencyMix     |    80.72   |    78.63   |    78.77   |    77.51    |    84.35   |
| AttentiveMix+   |    81.69   |    81.53   |    80.54   |    79.60    |    84.34   |
| FMix*           |    79.87   |    78.99   |    79.02   |    78.24    |    84.21   |
| PuzzleMix       |    81.69   |    82.84   |    82.85   |    82.93    |    85.02   |
| Co-Mixup📖     |    81.73   |    82.88   |    82.91   |    82.97    |    85.05   |
| ResizeMix*      |    79.56   |    79.78   |    80.35   |    79.73    |    84.87   |
| AlignMix📖     |            |            |            |             |            |
| AutoMix*        |    82.84   |    83.32   |    83.64   |    83.80    |    85.18   |
| SAMix*          |    83.81   |    84.27   |    84.42   |    84.31    |    85.50   |


## Fine-grained and Scenic Classification Benchmarks

We further provide benchmarks on downstream classification scenarios. We report the median of top-1 accuracy in the last 5/10 training epochs for 100/200 epochs.

### Transfer Learning on Small-scale Datasets

**Note**

* These benchmarks follow transfer learning settings on fine-grained datasets. use PyTorch pre-trained models as initialization and train 200 epochs on CUB-200 and FGVC-Aircraft.
* Please run configs in `configs/classification/aircrafts/` and `configs/classification/cub200/`.

| Datasets    |  CUB-200  |   CUB-200  |  Aircraft |  Aircraft  |
|-------------|:---------:|:----------:|:---------:|:----------:|
| Backbones   | ResNet-18 | ResNeXt-50 | ResNet-18 | ResNeXt-50 |
| Vanilla     |   77.68   |    83.01   |   80.23   |    85.10   |
| MixUp       |   78.39   |    84.58   |   79.52   |    85.18   |
| CutMix      |   78.40   |    85.68   |   78.84   |    84.55   |
| ManifoldMix |   79.76   |    86.38   |   80.68   |    86.60   |
| SaliencyMix |   77.95   |    83.29   |   80.02   |    84.31   |
| FMix*       |   77.28   |    84.06   |   79.36   |    86.23   |
| PuzzleMix   |   78.63   |    84.51   |   80.76   |    86.23   |
| ResizeMix*  |   78.50   |    84.77   |   78.10   |    84.08   |
| AutoMix*    |   79.87   |    86.56   |   81.37   |    86.72   |
| SAMix*      |   81.11   |    86.83   |   82.15   |    86.80   |

### Large-scale Datasets

**Note**

* These benchmarks largely based on PyTorch-style ImageNet-1k training settings, training 100 epochs from stretch on iNat2017/2018 and Place205.
* Please run configs in `configs/classification/inaturalist2017/`, `configs/classification/inaturalist2018/`, and `configs/classification/place205/`.

| Datasets    |  iNat2017 |   iNat2017  |  iNat2018 |   iNat2018  |
| ----------- | :-------: | :---------: | :-------: | :---------: |
| Backbones   | ResNet-50 | ResNeXt-101 | ResNet-50 | ResNeXt-101 |
| Vanilla     |   60.23   |    63.70    |   62.53   |    66.94    |
| MixUp       |   61.22   |    66.27    |   62.69   |    67.56    |
| CutMix      |   62.34   |    67.59    |   63.91   |    69.75    |
| ManifoldMix |   61.47   |    66.08    |   63.46   |    69.30    |
| SaliencyMix |   62.51   |    67.20    |   64.27   |    70.01    |
| FMix*       |   61.90   |    66.64    |   63.71   |    69.46    |
| PuzzleMix   |   62.66   |    67.72    |   64.36   |    70.12    |
| ResizeMix*  |   62.29   |    66.82    |   64.12   |    69.30    |
| AutoMix*    |   63.08   |    68.03    |   64.73   |    70.49    |
| SAMix*      |   63.32   |    68.26    |   64.84   |    70.54    |

| Datasets    |  Place205 |  Place205 |
| ----------- | :-------: | :-------: |
| Backbones   | ResNet-18 | ResNet-50 |
| Vanilla     |   59.63   |   63.10   |
| MixUp       |   59.33   |   63.01   |
| CutMix      |   59.21   |   63.75   |
| ManifoldMix |   59.46   |   63.23   |
| SaliencyMix |   59.50   |   63.33   |
| FMix*       |   59.51   |   63.63   |
| PuzzleMix   |   59.62   |   63.91   |
| ResizeMix*  |   59.66   |   63.88   |
| AutoMix*    |   59.74   |   64.06   |
| SAMix*      |   59.86   |   64.27   |
