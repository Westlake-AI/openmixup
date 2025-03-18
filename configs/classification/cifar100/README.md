# Mixup Classification Benchmark on CIFAR-100

> [Learning multiple layers of features from tiny images](http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

## Abstract

Groups at MIT and NYU have collected a dataset of millions of tiny colour images from the web. It is, in principle, an excellent dataset for unsupervised training of deep generative models, but previous researchers who have tried this have found it dicult to learn a good set of lters from the images. We show how to train a multi-layer generative model that learns to extract meaningful features which resemble those found in the human visual cortex. Using a novel parallelization algorithm to distribute the work among multiple machines connected on a network, we show how training such a model can be done in reasonable time. A second problematic aspect of the tiny images dataset is that there are no reliable class labels which makes it hard to use for object recognition experiments. We created two sets of reliable labels. The CIFAR-10 set has 6000 examples of each of 10 classes and the CIFAR-100 set has 600 examples of each of 100 non-overlapping classes. Using these labels, we show that object recognition is signicantly improved by pre-training a layer of features on a large set of unlabeled tiny images.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/189416919-54336ac1-cb8c-4df1-9929-aea6cd30c3b3.png" width="75%"/>
</div>

## Results and models

### Getting Started

* You can start training and evaluating with a config file. An example with a single GPU,
  ```shell
  python -c "import torchvision; torchvision.datasets.CIFAR100(download=True, root='data/cifar100');"
  CUDA_VISIBLE_DEVICES=1 PORT=29001 bash tools/dist_train.sh ${CONFIG_FILE} 1
  ```
* Please refer to config files for experiment details: [various mixups](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/), [AutoMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/automix/), [SAMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/samix/). As for config files of various mixups, please modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts. Here is an example of using Mixup and CutMix with switching probabilities of $\{0.4, 0.6\}$ based on [base_config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/basic/r18_mixups_CE_none.py).
  ```python
  model = dict(
      alpha=[0.1, 1],  # list of alpha
      mix_mode=["mixup", "cutmix"],  # list of chosen mixup modes
      mix_prob=[0.4, 0.6],  # list of applying probs (sum=1), `None` for random applying
      mix_repeat=1,  # times of repeating mixups in each iteration
  )
  ```

<<<<<<< HEAD
### Mixup Benchmark on CIFAR-100
=======
### Classical Mixup Benchmark on CIFAR-100
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)

**Setup**

* This benchmark using CIFAR varient of ResNet architectures train 200, 400, 800, 1200 epochs on [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). The training and testing image size is 32 and we search $\alpha$ in $Beta(\alpha, \alpha)$ for all compared methods.
<<<<<<< HEAD
* As for mixup variants requiring some special components, we provide examples based on ResNet-18: [AttentiveMix+](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/basic/r18_attentivemix_CE_none.py) and [PuzzleMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/basic/r18_puzzlemix_CE_soft.py).
=======
* View config files of general Mixup methods in [mixups/basic](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/basic/) and the [DecoupleMix](https://arxiv.org/abs/2203.10761) variants in [mixups/decouple](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/decouple/). As for mixup variants requiring some special components, we provide examples based on ResNet-18: [AttentiveMix+](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/basic/r18_attentivemix_CE_none.py) and [PuzzleMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/basic/r18_puzzlemix_CE_soft.py).
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
* The **median** of top-1 accuracy in the last 10 training epochs is reported for ResNet-18, ResNeXt-50 (32x4d), and Wide-ResNet-28-8. Notice that 📖 denotes original results reproduced by official implementations.

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

<<<<<<< HEAD
### ViTs Benchmark on CIFAR-100

On updating!
=======
### ViTs' Mixup Benchmark on CIFAR-100

**Setup**

* Since the original resolutions of CIFAR-100 are too small for ViTs, we resize the input images to $224\times 224$ (training and testing) while not modifying the ViT architectures. We use the CIFAR version of ConvNeXt-Tiny with $32\times 32$ as the input resolution. This benchmark uses DeiT setup and trains the model for 200 and 600 epochs with a batch size of 100 on CIFAR-100. The basic learning rate of DeiT/ConvNeXt and Swin are $1e-3$ and $5e-4$, which is the optimal setup in our experiments. We search and report $\alpha$ in $Beta(\alpha, \alpha)$ for all compared methods. View config files in [mixups/vits](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/vits/).
* The **best** of top-1 accuracy in the last 10 training epochs is reported for ViT architectures. We released the trained models and logs in [vits-mix-cifar100-weights](https://github.com/Westlake-AI/openmixup/releases/tag/vits-mix-cifar100-weights).

| Backbones     |  $Beta$  | DEiT-S(/16) | DEiT-S(/16) | DEiT-S(/16) |   DEiT-S(/16)  |
|---------------|:--------:|:-----------:|:-----------:|:-----------:|:--------------:|
| Epoch         | $\alpha$ |  200 epochs |  600 epochs |  Memory (G) | 600ep Time (h) |
| Vanilla       |     -    |    65.81    |    68.50    |     8.1     |       27       |
| MixUp         |    0.8   |    69.98    |    76.35    |     8.2     |       27       |
| CutMix        |     2    |    74.12    |    79.54    |     8.2     |       27       |
| DeiT          |   0.8,1  |    75.92    |    79.38    |     8.2     |       27       |
| SmoothMix     |    0.2   |    67.54    |    80.25    |     8.2     |       27       |
| SaliencyMix   |    0.2   |    69.78    |    76.60    |     8.2     |       27       |
| AttentiveMix+ |     2    |    75.98    |    80.33    |     8.3     |       35       |
| FMix*         |     1    |    70.41    |    74.31    |     8.2     |       27       |
| GridMix       |     1    |    68.86    |    74.96    |     8.2     |       27       |
| PuzzleMix     |     2    |    73.60    |    81.01    |     8.3     |       35       |
| ResizeMix*    |     1    |    68.45    |    71.95    |     8.2     |       27       |
| TransMix      |   0.8,1  |    76.17    |    79.33    |     8.4     |       28       |
| AutoMix       |     2    |    76.24    |    80.91    |     18.2    |       59       |
| SAMix*        |     2    |    77.94    |    82.49    |     21.3    |       58       |

| Backbones     |  $Beta$  |   Swin-T   |   Swin-T   |   Swin-T   |     Swin-T     |
|---------------|:--------:|:----------:|:----------:|:----------:|:--------------:|
| Epoch         | $\alpha$ | 200 epochs | 600 epochs | Memory (G) | 600ep Time (h) |
| Vanilla       |     -    |    78.41   |    81.29   |    11.4    |       36       |
| MixUp         |    0.8   |    76.78   |    83.67   |    11.4    |       36       |
| CutMix        |     2    |    80.64   |    83.38   |    11.4    |       36       |
| DeiT          |   0.8,1  |    81.25   |    84.82   |    11.4    |       36       |
| SmoothMix     |    0.2   |    66.69   |    81.18   |    11.4    |       36       |
| SaliencyMix   |    0.2   |    80.40   |    82.58   |    11.4    |       36       |
| AttentiveMix+ |     2    |    81.13   |    83.69   |    11.5    |       43       |
| FMix*         |     1    |    80.72   |    82.82   |    11.4    |       36       |
| GridMix       |     1    |    78.54   |    80.79   |    11.4    |       36       |
| PuzzleMix     |     2    |    80.33   |    83.13   |    11.5    |       45       |
| ResizeMix     |     1    |    80.16   |    82.36   |    11.4    |       36       |
| TransMix      |   0.8,1  |    81.33   |            |    11.5    |       37       |
| AutoMix       |     2    |    82.67   |            |    29.2    |       75       |
| SAMix*        |     2    |    82.62   |            |    29.3    |       75       |

| Backbones     |  $Beta$  | ConvNeXt-T | ConvNeXt-T | ConvNeXt-T |   ConvNeXt-T   |
|---------------|:--------:|:----------:|:----------:|:----------:|:--------------:|
| Epoch         | $\alpha$ | 200 epochs | 600 epochs | Memory (G) | 600ep Time (h) |
| Vanilla       |     -    |    78.70   |    80.65   |     4.2    |       10       |
| MixUp         |    0.8   |    81.13   |    83.08   |     4.2    |       10       |
| CutMix        |     2    |    82.46   |    83.20   |     4.2    |       10       |
| DeiT          |   0.8,1  |    83.09   |    84.12   |     4.2    |       10       |
| ManifoldMix   |     2    |    82.06   |    83.94   |     4.2    |       10       |
| SmoothMix     |    0.2   |    78.87   |    81.31   |     4.2    |       10       |
| SaliencyMix   |    0.2   |    82.82   |    83.03   |     4.2    |       10       |
| AttentiveMix+ |     2    |    82.59   |    83.04   |     4.3    |       14       |
| FMix*         |     1    |    81.79   |    82.29   |     4.2    |       10       |
| GridMix       |     1    |    79.53   |    79.66   |     4.2    |       10       |
| PuzzleMix     |     2    |    82.29   |    84.17   |     4.3    |       53       |
| ResizeMix     |     1    |    82.53   |    82.91   |     4.2    |       10       |
| AutoMix       |     2    |    83.30   |    84.79   |    10.2    |       28       |
| SAMix*        |     2    |    83.56   |    84.98   |    10.3    |       29       |
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)

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
