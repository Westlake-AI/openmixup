# Mixup Classification Benchmark on ImageNet

> [ImageNet Large Scale Visual Recognition Challenge](https://dl.acm.org/doi/10.1145/3065386)

## Abstract

The ImageNet Large Scale Visual Recognition Challenge is a benchmark in object category classification and detection on hundreds of object categories and millions of images. The challenge has been run annually from 2010 to present, attracting participation from more than fifty institutions. This paper describes the creation of this benchmark dataset and the advances in object recognition that have been possible as a result. We discuss the challenges of collecting large-scale ground truth annotation, highlight key breakthroughs in categorical object recognition, provide a detailed analysis of the current state of the field of large-scale image classification and object detection, and compare the state-of-the-art computer vision accuracy with human accuracy. We conclude with lessons learned in the five years of the challenge, and propose future directions and improvements.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/204880775-7562bf12-2f10-4ee9-9cac-90ceac13f098.png" width="100%"/>
</div>

## Results and models

We provide three popular benchmarks on ImageNet-1k based on various network architectures. We also provide results on Tiny-ImageNet for fast experiments. The **median** of top-1 accuracy in the last 5/10 training epochs for 100/300 epochs is reported for ResNet variants, and the **best** top-1 accuracy is reported for Transformer architectures.

### **PyTorch-style Training Settings on ImageNet-1k**

These benchmarks follow [PyTorch-style](https://arxiv.org/abs/2110.00476) settings, training 100 and 300 epochs from stretch based on ResNet variants on [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/).

**Note**

* Please refer to config files for experiment details: [various mixups](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/basic/), [AutoMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix/basic/), [SAMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix/basic/). As for config files of [various mixups](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/basic/), please modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts.
* Since ResNet-18 might be under-fitted on ImageNet-1k, we adopt $\alpha=0.2$ for some cutting-based mixups (CutMix, SaliencyMix, FMix, ResizeMix) based on ResNet-18.
* Notice that 📖 denotes original results reproduced by official implementations.

| Backbones     |  $Beta$  |  ResNet-18 |  ResNet-34 |  ResNet-50 | ResNet-101 | ResNeXt-101 |
|---------------|:--------:|:----------:|:----------:|:----------:|:----------:|:-----------:|
| Epochs        | $\alpha$ | 100 epochs | 100 epochs | 100 epochs | 100 epochs |  100 epochs |
| Vanilla       |     -    |    70.04   |    73.85   |    76.83   |    78.18   |    78.71    |
| MixUp         |    0.2   |    69.98   |    73.97   |    77.12   |    78.97   |    79.98    |
| CutMix        |     1    |    68.95   |    73.58   |    77.17   |    78.96   |    80.42    |
| ManifoldMix   |    0.2   |    69.98   |    73.98   |    77.01   |    79.02   |    79.93    |
| SaliencyMix   |     1    |    69.16   |    73.56   |    77.14   |    79.32   |    80.27    |
| AttentiveMix+ |     2    |    68.57   |      -     |    77.28   |      -     |      -      |
| FMix*         |     1    |    69.96   |    74.08   |    77.19   |    79.09   |    80.06    |
| PuzzleMix     |     1    |    70.12   |    74.26   |    77.54   |    79.43   |    80.53    |
| Co-Mixup📖     |     2    |      -     |      -     |    77.60   |      -     |      -      |
| SuperMix📖     |     2    |      -     |      -     |    77.63   |      -     |      -      |
| ResizeMix*    |     1    |    69.50   |    73.88   |    77.42   |    79.27   |    80.55    |
| AlignMix📖     |     2    |      -     |      -     |    78.00   |      -     |      -      |
| Grafting📖     |     1    |      -     |      -     |    77.74   |      -     |      -      |
| AutoMix       |     2    |    70.50   |    74.52   |    77.91   |    79.87   |    80.89    |
| SAMix*        |     2    |    70.83   |    74.95   |    78.06   |    80.05   |    80.98    |

| Backbones   |  $Beta$  |  ResNet-18 |  ResNet-34 |  ResNet-50 | ResNet-101 |
|-------------|:--------:|:----------:|:----------:|:----------:|:----------:|
| Epochs      | $\alpha$ | 300 epochs | 300 epochs | 300 epochs | 300 epochs |
| Vanilla     |     -    |    71.83   |    75.29   |    77.35   |    78.91   |
| MixUp       |    0.2   |    71.72   |    75.73   |    78.44   |    80.60   |
| CutMix      |     1    |    71.01   |    75.16   |    78.69   |    80.59   |
| ManifoldMix |    0.2   |    71.73   |    75.44   |    78.21   |    80.64   |
| SaliencyMix |     1    |    70.21   |    75.01   |    78.46   |    80.45   |
| FMix*       |     1    |    70.30   |    75.12   |    78.51   |    80.20   |
| PuzzleMix   |     1    |    71.64   |    75.84   |    78.86   |    80.67   |
| ResizeMix*  |     1    |    71.32   |    75.64   |    78.91   |    80.52   |
| AlignMix📖   |     2    |      -     |      -     |    79.32   |      -     |
| AutoMix     |     2    |    72.05   |    76.10   |    79.25   |    80.98   |
| SAMix*      |     2    |    72.27   |    76.28   |    79.39   |    81.10   |

### **Timm RSB A2/A3 Training Settings on ImageNet-1k**

These benchmarks follow [timm](https://github.com/rwightman/pytorch-image-models) [RSB A2/A3](https://arxiv.org/abs/2110.00476) settings based on ResNet-50, EfficientNet-B0, and MobileNet.V2. Training 300/100 epochs with the BCE loss on ImageNet-1k, RSB A3 is a fast training setting while RSB A2 can exploit the full representation ability of ConvNets.

**Note**

* Please refer to config files for experiment details: [RSB A3](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/rsb_a3/) and [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/rsb_a2/). You can modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts for various mixups.
* Notice that the [RSB](https://arxiv.org/abs/2110.00476) settings employ Mixup with $\alpha=0.1$ and CutMix with $\alpha=1.0$. We report the **median** of top-1 accuracy in the last 5/10 training epochs for 100/300 epochs.

| Backbones     |  $Beta$  | ResNet-50 | ResNet-50 | Eff-B0 | Eff-B0 | Mob.V2 1x | Mob.V2 1x |
|---------------|:--------:|:---------:|:---------:|:------:|:------:|:---------:|:---------:|
| Settings      | $\alpha$ |     A3    |     A2    |   A3   |   A2   |     A3    |     A2    |
| RSB           |  0.1, 1  |   78.08   |   79.80   |  74.02 |  77.26 |   69.86   |   72.87   |
| MixUp         |    0.2   |   77.66   |     -     |  73.87 |  77.19 |   70.17   |   72.78   |
| CutMix        |    0.2   |   77.62   |   79.38   |  73.46 |  77.24 |   69.62   |   72.23   |
| ManifoldMix   |    0.2   |   77.78   |   79.47   |  73.83 |  77.22 |   70.05   |   72.34   |
| AttentiveMix+ |     2    |   77.46   |   79.34   |  72.16 |  75.95 |   67.32   |   70.30   |
| SaliencyMix   |    0.2   |   77.93   |   79.42   |  73.42 |  77.67 |   69.69   |   72.07   |
| FMix*         |    0.2   |   77.76   |   79.05   |  73.71 |  77.33 |   70.10   |   72.79   |
| PuzzleMix     |     1    |   78.02   |   79.78   |  74.10 |  77.35 |   70.04   |   72.85   |
| ResizeMix*    |     1    |   77.85   |   79.74   |  73.67 |  77.27 |   69.94   |   72.50   |
| AutoMix       |     2    |   78.44   |     -     |  74.61 |  77.58 |   71.16   |   73.19   |
| SAMix         |     2    |   78.64   |     -     |  75.28 |  77.69 |   71.24   |   73.42   |

### **DeiT Training Settings with ViTs on ImageNet-1k**

Since recently proposed transformer-based architectures adopt mixups as parts of essential augmentations, these benchmarks follow [DeiT](https://arxiv.org/abs/2012.12877) settings based on DeiT-Small, Swin-Tiny, and ConvNeXt-Tiny on ImageNet-1k.

**Note**

* Please refer to config files of various mixups for experiment details: [DeiT](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/deit/), [PVT](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/pvt), [Swin](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/swin/), [ConvNeXt](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/convnext/), [MogaNet](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mixups/moganet/). You can modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts for various mixups.
* Notice that the [DeiT](https://arxiv.org/abs/2012.12877) setting employs Mixup with $\alpha=0.8$ and CutMix with $\alpha=1.0$.
* Notice that the performances of transformer-based architectures are more difficult to reproduce than ResNet variants, and the mean of the **best** performance in 3 trials is reported as their original paper. Notice that 📖 denotes original results reproduced by official implementations.

| Methods       | $\alpha$ | DeiT-T | DeiT-S | PVT-T | Swin-T | ConvNeXt-T | MogaNet-T |
|---------------|:--------:|:------:|:------:|:-----:|:------:|:----------:|:---------:|
| Vanilla       |     -    |        |  75.66 |       |  80.21 |    79.22   |   79.25   |
| DeiT          |  0.8, 1  |  74.50 |  79.80 | 75.10 |  81.20 |    82.10   |   79.02   |
| MixUp         |    0.2   |  74.69 |  77.72 | 75.24 |  81.01 |    80.88   |   79.29   |
| CutMix        |    0.2   |  73.82 |  80.13 | 75.53 |  81.23 |    81.57   |   78.37   |
| ManifoldMix   |    0.2   |    -   |    -   |   -   |    -   |    80.57   |   79.07   |
| AttentiveMix+ |     2    |  74.07 |  80.32 | 74.98 |  81.29 |    81.14   |   77.53   |
| SaliencyMix   |    0.2   |        |  79.88 | 75.71 |  81.37 |    81.33   |   78.74   |
| PuzzleMix     |     1    |  73.85 |  80.45 | 75.48 |  81.47 |    81.48   |   78.12   |
| FMix*         |    0.2   |  74.41 |  77.37 | 75.28 |  79.60 |    81.04   |   79.05   |
| ResizeMix*    |     1    |  74.79 |  78.61 | 76.05 |  81.36 |    81.64   |   78.77   |
| TransMix📖     |  0.8, 1  |  72.92 |  80.70 | 75.50 |  81.80 |      -     |     -     |
| TokenMix📖     |  0.8, 1  |  75.31 |  80.80 | 75.60 |  81.60 |      -     |     -     |
| AutoMix       |     2    |  75.52 |  80.78 | 76.38 |  81.80 |    82.28   |   79.43   |
| SAMix*        |     2    |        |  80.94 | 76.60 |  81.87 |    82.35   |           |

We summarize mixup benchmarks in [Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md).


## Citation

Please refer to the original paper of [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and [AutoMix](https://arxiv.org/abs/2103.13027) for details.

```bibtex
@article{IJCV2015ImageNet,
  title={ImageNet Large Scale Visual Recognition Challenge},
  author={Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael S. Bernstein and Alexander C. Berg and Li Fei-Fei},
  journal={International Journal of Computer Vision},
  year={2015},
  volume={115},
  pages={211-252}
}
```
