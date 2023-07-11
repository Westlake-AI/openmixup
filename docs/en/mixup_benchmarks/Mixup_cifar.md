# Mixup CIFAR-10/100 Benchmarks

**OpenMixup provides mixup benchmarks on supervised learning on various tasks. Config files and experiment results are available, and pre-trained models and training logs are updating. Moreover, more advanced mixup variants will be supported in the future. Issues and PRs are welcome!**

Now, we have supported 13 popular mixup methods! Notice that * denotes open-source arXiv pre-prints reproduced by us, and 📖 denotes original results reproduced by official implementations. We modified the original AttentiveMix by using pre-trained R-18 (or R-50) and sampling $\lambda$ from $\Beta(\alpha,8)$ as *AttentiveMix+*, which yields better performances.

**Note**

* We summarize benchmark results in Markdown tables. You can convert them into other formats (e.g., LaTeX) with [online tools](https://www.tablesgenerator.com/markdown_tables).
* As for evaluation, you can test pre-trained models with `tools/dist_test.sh`, and then you can extract experiment results (from JSON files) by tools in `openmixup/tools/summary/`. An example with 4 GPUs evaluation and summarization is as follows:
  ```shell
  CUDA_VISIBLE_DEVICES=1,2,3,4 bash tools/dist_test.sh ${CONFIG_FILE} 4 ${PATH_TO_MODEL}
  python tools/summary/find_val_max_3times_average.py ${PATH_TO_JSON_LOG} head0_top1-head0_top5
  ```

<details open>
<summary>Supported sample mixing policies</summary>

- [X] [Mixup [ICLR'2018]](https://arxiv.org/abs/1710.09412)
- [X] [CutMix [ICCV'2019]](https://arxiv.org/abs/1905.04899)
- [X] [ManifoldMix [ICML'2019]](https://arxiv.org/abs/1806.05236)
- [X] [FMix [ArXiv'2020]](https://arxiv.org/abs/2002.12047)
- [X] [AttentiveMix [ICASSP'2020]](https://arxiv.org/abs/2003.13048)
- [X] [SmoothMix [CVPRW'2020]](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.pdf)
- [X] [SaliencyMix [ICLR'2021]](https://arxiv.org/abs/1710.09412)
- [X] [PuzzleMix [ICML'2020]](https://arxiv.org/abs/2009.06962)
- [ ] [Co-Mixup [ICLR'2021]](https://openreview.net/forum?id=gvxJzw8kW4b)
- [X] [GridMix [Pattern Recognition'2021]](https://www.sciencedirect.com/science/article/pii/S0031320320303976)
- [ ] [SuperMix [CVPR'2021]](https://arxiv.org/abs/2003.05034)
- [X] [ResizeMix [ArXiv'2020]](https://arxiv.org/abs/2012.11101)
- [X] [AlignMix [CVPR'2022]](https://arxiv.org/abs/2103.15375)
- [X] [AutoMix [ECCV'2022]](https://arxiv.org/abs/2103.13027)
- [X] [SAMix [ArXiv'2021]](https://arxiv.org/abs/2111.15454)
- [ ] [RecursiveMix [ArXiv'2022]](https://arxiv.org/abs/2203.06844)

</details>

<details open>
<summary>Supported label mixing policies</summary>

- [ ] [Saliency Grafting [AAAI'2022]](https://arxiv.org/abs/2112.08796)
- [X] [TransMix [CVPR'2022]](https://arxiv.org/abs/2111.09833)
- [X] [DecoupleMix [ArXiv'2022]](https://arxiv.org/abs/2203.10761)
- [ ] [TokenMix [ECCV'2022]](https://arxiv.org/abs/2207.08409)

</details>

## CIFAR-10/100 Benchmarks

CIFAR benchmarks based on ResNet variants. We report the **median** of top-1 accuracy in the last 10 training epochs.

### **CIFAR-10 Dataset**

These benchmarks follow CutMix settings, training 200/400/800/1200 epochs from stretch based on the CIFAR version of ResNet variants on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

**Note**

* Please refer to config files of [CIFAR-10](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar10/) for experiment details. You can modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts for various mixups.
* Notice that 📖 denotes original results reproduced by official implementations.

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
| AlignMix📖    |     2     |      -     |      -     |      -     |    97.05    |
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

<p align="right">(<a href="#top">back to top</a>)</p>

### **CIFAR-100 Dataset**

#### Classical Mixup Benchmark on CIFAR-100

These benchmarks follow CutMix settings, training 200/400/800/1200 epochs from stretch based on the CIFAR version of ResNet variants on [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). When adopting ResNeXt-50 (32x4d) as the backbone, we use `wd=5e-4` for cutting-based methods (CutMix, AttributeMix+, SaliencyMix, FMix, ResizeMix) for better performances.

**Note**

* Please refer to config files for experiment details: [various mixups](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/), [AutoMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/automix/), [SAMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/samix/), [DecoupleMix](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/decouple/). As for config files of [various mixups](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/), please modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts.
* Notice that 📖 denotes original results reproduced by official implementations.

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
| AlignMix📖    |     2     |      -     |      -     |      -     |    81.71    |
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

#### ViTs' Mixup Benchmark on CIFAR-100

**Setup**

* Since the original resolutions of CIFAR-100 are too small for ViTs, we resize the input images to $224\times 224$ (training and testing) while not modifying the ViT architectures. This benchmark uses DeiT setup and trains the model for 200 epochs with a batch size of 100 on CIFAR-100. The basic learning rate of DeiT and Swin are $1e-3$ and $5e-4$, which is the optimal setup in our experiments. We search and report $\alpha$ in $Beta(\alpha, \alpha)$ for all compared methods. View config files in [mixups/vits](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cifar100/mixups/vits/).
* The **best** of top-1 accuracy in the last 10 training epochs is reported for ViT architectures. Notice that 📖 denotes original results reproduced by official implementations.

| Backbones     |  $Beta$  | DEiT-S(/16) |   Swin-T   |
|---------------|:--------:|:-----------:|:----------:|
| Epoch         | $\alpha$ |  200 epochs | 200 epochs |
| Vanilla       |     -    |    65.81    |    78.41   |
| MixUp         |    0.8   |    69.98    |    76.78   |
| CutMix        |     2    |    74.12    |    80.64   |
| DeiT          |   0.8,1  |    75.92    |    81.25   |
| SmoothMix     |    0.2   |    67.54    |    66.69   |
| SaliencyMix   |    0.2   |    69.78    |    80.40   |
| AttentiveMix+ |     2    |    75.98    |    81.13   |
| FMix*         |     1    |    70.41    |    80.72   |
| GridMix       |     1    |    68.86    |    78.54   |
| PuzzleMix     |     2    |    73.60    |    80.33   |
| ResizeMix*    |     1    |    68.45    |    80.16   |
| TransMix      |   0.8,1  |    76.17    |      -     |
| AutoMix       |     2    |    74.87    |    82.67   |
| SAMix*        |     2    |             |            |

<p align="right">(<a href="#top">back to top</a>)</p>