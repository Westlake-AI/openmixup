# Mixup Classification Benchmark on CIFAR-100

> [Learning multiple layers of features from tiny images](http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

## Abstract

Groups at MIT and NYU have collected a dataset of millions of tiny colour images from the web. It is, in principle, an excellent dataset for unsupervised training of deep generative models, but previous researchers who have tried this have found it dicult to learn a good set of lters from the images. We show how to train a multi-layer generative model that learns to extract meaningful features which resemble those found in the human visual cortex. Using a novel parallelization algorithm to distribute the work among multiple machines connected on a network, we show how training such a model can be done in reasonable time. A second problematic aspect of the tiny images dataset is that there are no reliable class labels which makes it hard to use for object recognition experiments. We created two sets of reliable labels. The CIFAR-10 set has 6000 examples of each of 10 classes and the CIFAR-100 set has 600 examples of each of 100 non-overlapping classes. Using these labels, we show that object recognition is signicantly improved by pre-training a layer of features on a large set of unlabeled tiny images.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/189416919-54336ac1-cb8c-4df1-9929-aea6cd30c3b3.png" width="90%"/>
</div>

## Results and models

* This benchmark using CIFAR varient of ResNet architectures train 200, 400, 800, 1200 epochs on [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). The training and testing image size is 32 and we search $\alpha$ in $Beta(\alpha, \alpha)$ for all compared methods.
* Please refer to config files for experiment details: [various mixups](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/), [AutoMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/automix/), [SAMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/samix/). As for config files of various mixups, please modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts.
* The **median** of top-1 accuracy in the last 10 training epochs is reported for ResNet-18, ResNeXt-50 (32x4d), and Wide-ResNet-28-8. Notice that 📖 denotes original results reproduced by official implementations.

### CIFAR-100

| Backbones     |  $Beta$   |  ResNet-18 |  ResNet-18 |  ResNet-18 |  ResNet-18  |
|---------------|:---------:|:----------:|:----------:|:----------:|:-----------:|
| Epoch         |  $\alpha$ | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
| Vanilla       |     -     |    76.42   |    77.73   |    78.04   |    78.55    |
| MixUp         |     1     |    78.52   |    79.34   |    79.12   |    79.24    |
| CutMix        |    0.2    |    79.45   |    79.58   |    78.17   |    78.29    |
| ManifoldMix   |     2     |    79.18   |    80.18   |    80.35   |    80.21    |
| SmoothMix     |    0.2    |    77.90   |    78.77   |    78.69   |    78.38    |
| SaliencyMix   |    0.2    |    79.75   |    79.64   |    79.12   |    77.66    |
| AttentiveMix+ |     2     |    79.62   |    80.14   |    78.91   |    78.41    |
| FMix*         |    0.2    |    78.91   |    79.91   |    79.69   |    79.50    |
| PuzzleMix     |     1     |    79.96   |    80.82   |    81.13   |    81.10    |
| Co-Mixup📖    |     2     |    80.01   |    80.87   |    81.17   |    81.18    |
| GridMix       |    0.2    |    78.23   |    78.60   |    78.72   |    77.58    |
| ResizeMix*    |     1     |    79.56   |    79.19   |    80.01   |    79.23    |
| AutoMix       |     2     |    80.12   |    81.78   |    82.04   |    81.95    |
| SAMix*        |     2     |    81.21   |    81.97   |    82.30   |    82.41    |

| Backbones     |  $Beta$  | ResNeXt-50 | ResNeXt-50 | ResNeXt-50 |  ResNeXt-50 |  WRN-28-8  |
|---------------|:--------:|:----------:|:----------:|:----------:|:-----------:|:----------:|
| Epoch         | $\alpha$ | 200 epochs | 400 epochs | 800 epochs | 1200 epochs | 400 epochs |
| Vanilla       |     -    |    79.37   |    80.24   |    81.09   |    81.32    |    81.63   |
| MixUp         |     1    |    81.18   |    82.54   |    82.10   |    81.77    |    82.82   |
| CutMix        |    0.2   |    81.52   |    78.52   |    78.32   |    77.17    |    84.45   |
| ManifoldMix   |     2    |    81.59   |    82.56   |    82.88   |    83.28    |    83.24   |
| SmoothMix     |    0.2   |    80.68   |    79.56   |    78.95   |    77.88    |    82.09   |
| SaliencyMix   |    0.2   |    80.72   |    78.63   |    78.77   |    77.51    |    84.35   |
| AttentiveMix+ |     2    |    81.69   |    81.53   |    80.54   |    79.60    |    84.34   |
| FMix*         |    0.2   |    79.87   |    78.99   |    79.02   |    78.24    |    84.21   |
| PuzzleMix     |     1    |    81.69   |    82.84   |    82.85   |    82.93    |    85.02   |
| Co-Mixup📖    |     2    |    81.73   |    82.88   |    82.91   |    82.97    |    85.05   |
| GridMix       |    0.2   |    81.11   |    79.80   |    78.90   |    76.11    |    84.24   |
| ResizeMix*    |     1    |    79.56   |    79.78   |    80.35   |    79.73    |    84.87   |
| AutoMix       |     2    |    82.84   |    83.32   |    83.64   |    83.80    |    85.18   |
| SAMix*        |     2    |    83.81   |    84.27   |    84.42   |    84.31    |    85.50   |

We summarize mixup benchmarks in [Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md).


## Citation

Please refer to the original paper of [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and [AutoMix](https://arxiv.org/abs/2103.13027) for details.

```bibtex
@article{Krizhevsky2009Cifar,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey and others},
  year={2009},
  publisher={Citeseer}
}
```
```bibtex
@misc{eccv2022automix,
  title={AutoMix: Unveiling the Power of Mixup for Stronger Classifiers},
  author={Zicheng Liu and Siyuan Li and Di Wu and Zhiyuan Chen and Lirong Wu and Jianzhu Guo and Stan Z. Li},
  year={2021},
  eprint={2103.13027},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
