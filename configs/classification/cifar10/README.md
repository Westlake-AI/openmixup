# Mixup Classification Benchmark on CIFAR-10

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
  python -c "import torchvision; torchvision.datasets.CIFAR10(download=True, root='data/cifar10');"
  CUDA_VISIBLE_DEVICES=1 PORT=29001 bash tools/dist_train.sh ${CONFIG_FILE} 1
  ```
* Please refer to [configs](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar10/mixups/basic) for experiment details. You can modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts. Here is an example of using Mixup and CutMix with switching probabilities of $\{0.4, 0.6\}$ based on [base_config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar10/mixups/basic/r18_mixups_CE_none.py).
  ```python
  model = dict(
      alpha=[0.1, 1],  # list of alpha
      mix_mode=["mixup", "cutmix"],  # list of chosen mixup modes
      mix_prob=[0.4, 0.6],  # list of applying probs (sum=1), `None` for random applying
      mix_repeat=1,  # times of repeating mixups in each iteration
  )
  ```

### CIFAR-10

**Setup**

* This benchmark using CIFAR varient of ResNet architectures train 200, 400, 800, 1200 epochs on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). The training and testing image size is 32 and we search $\alpha$ in $Beta(\alpha, \alpha)$ for all compared methods.
<<<<<<< HEAD
* As for mixup variants requiring some special components, we provide examples based on ResNet-18: [AttentiveMix+](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar10/mixups/basic/r18_attentivemix_CE_none.py) and [PuzzleMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar10/mixups/basic/r18_puzzlemix_CE_soft.py).
=======
* View config files of general Mixup methods in [mixups/basic](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar10/mixups/basic/). As for mixup variants requiring some special components, we provide examples based on ResNet-18: [AttentiveMix+](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar10/mixups/basic/r18_attentivemix_CE_none.py) and [PuzzleMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar10/mixups/basic/r18_puzzlemix_CE_soft.py).
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
* The **median** of top-1 accuracy in the last 10 training epochs is reported for ResNet-18 and ResNeXt-50 (32x4d).

| Backbones     |  $Beta$   |  ResNet-18 |  ResNet-18 |  ResNet-18 |  ResNet-18  |
|---------------|:---------:|:----------:|:----------:|:----------:|:-----------:|
| Epochs        |  $\alpha$ | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
| Vanilla       |     -     |    94.87   |    95.10   |    95.50   |    95.59    |
| MixUp         |     1     |    95.70   |    96.55   |    96.62   |    96.84    |
| CutMix        |    0.2    |    96.11   |    96.13   |    96.68   |    96.56    |
| ManifoldMix   |     2     |    96.04   |    96.57   |    96.71   |    97.02    |
| SmoothMix     |    0.5    |    95.29   |    95.88   |    96.17   |    96.17    |
| SaliencyMix   |    0.2    |    96.05   |    96.42   |    96.20   |    96.18    |
| AttentiveMix+ |     2     |    96.21   |    96.45   |    96.63   |    96.49    |
| FMix*         |    0.2    |    96.17   |    96.53   |    96.18   |    96.01    |
| PuzzleMix     |     1     |    96.42   |    96.87   |    97.10   |    97.13    |
| GridMix       |    0.2    |    95.89   |    96.33   |    96.56   |    96.58    |
| ResizeMix*    |     1     |    96.16   |    96.91   |    96.76   |    97.04    |
| AutoMix       |     2     |    96.59   |    97.08   |    97.34   |    97.30    |
| SAMix*        |     2     |    96.67   |    97.16   |    97.50   |    97.41    |

| Backbones     |   $Beta$   | ResNeXt-50 | ResNeXt-50 | ResNeXt-50 |  ResNeXt-50 |
|---------------|:----------:|:----------:|:----------:|:----------:|:-----------:|
| Epochs        |  $\alpha$  | 200 epochs | 400 epochs | 800 epochs | 1200 epochs |
| Vanilla       |      -     |    95.92   |    95.81   |    96.23   |    96.26    |
| MixUp         |      1     |    96.88   |    97.19   |    97.30   |    97.33    |
| CutMix        |     0.2    |    96.78   |    96.54   |    96.60   |    96.35    |
| ManifoldMix   |      2     |    96.97   |    97.39   |    97.33   |    97.36    |
| SmoothMix     |     0.2    |    95.87   |    96.37   |    96.49   |    96.77    |
| SaliencyMix   |     0.2    |    96.65   |    96.89   |    96.70   |    96.60    |
| AttentiveMix+ |      2     |    96.84   |    96.91   |    96.87   |    96.62    |
| FMix*         |     0.2    |    96.72   |    96.76   |    96.76   |    96.10    |
| PuzzleMix     |      1     |    97.05   |    97.24   |    97.37   |    97.34    |
| GridMix       |     0.2    |    97.18   |    97.30   |    96.40   |    95.79    |
| ResizeMix*    |      1     |    97.02   |    97.38   |    97.21   |    97.36    |
| AutoMix       |      2     |    97.19   |    97.42   |    97.65   |    97.51    |
| SAMix*        |      2     |    97.23   |    97.51   |    97.93   |    97.74    |

We summarize mixup benchmarks in [Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md).


## Citation

Please refer to the original paper of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [AutoMix](https://arxiv.org/abs/2103.13027) for details.

```bibtex
@article{Krizhevsky2009Cifar,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey and others},
  year={2009},
  publisher={Citeseer}
}
```
