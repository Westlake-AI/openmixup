# Mixup Fine-grained and Scenic Classification Benchmarks

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
- [ ] [TransMix [CVPR'2022]](https://arxiv.org/abs/2111.09833)
- [X] [DecoupleMix [ArXiv'2022]](https://arxiv.org/abs/2203.10761)
- [ ] [TokenMix [ECCV'2022]](https://arxiv.org/abs/2207.08409)

</details>

## Fine-grained and Scenic Classification Benchmarks

We further provide benchmarks on downstream classification scenarios. We report the **median** of top-1 accuracy in the last 5/10 training epochs for 100/200 epochs.

### **Transfer Learning on Small-scale Fine-grained Datasets**

These benchmarks follow transfer learning settings on fine-grained datasets, using PyTorch official pre-trained models as initialization and training 200 epochs on [CUB-200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/).

**Note**

* Please refer to config files for experiment details: [CUB-200](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cub200/mixups/basic) and [FGVC-Aircraft](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/aircrafts/mixups/basic). As for config files of various mixups, please modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts.

| Datasets    |  $Beta$  |  CUB-200  |   CUB-200  |  Aircraft |  Aircraft  |
|-------------|:--------:|:---------:|:----------:|:---------:|:----------:|
| Backbones   | $\alpha$ | ResNet-18 | ResNeXt-50 | ResNet-18 | ResNeXt-50 |
| Vanilla     |     -    |   77.68   |    83.01   |   80.23   |    85.10   |
| MixUp       |    0.2   |   78.39   |    84.58   |   79.52   |    85.18   |
| CutMix      |     1    |   78.40   |    85.68   |   78.84   |    84.55   |
| ManifoldMix |    0.5   |   79.76   |    86.38   |   80.68   |    86.60   |
| SaliencyMix |    0.2   |   77.95   |    83.29   |   80.02   |    84.31   |
| FMix*       |    0.2   |   77.28   |    84.06   |   79.36   |    86.23   |
| PuzzleMix   |     1    |   78.63   |    84.51   |   80.76   |    86.23   |
| ResizeMix*  |     1    |   78.50   |    84.77   |   78.10   |    84.08   |
| AutoMix     |     2    |   79.87   |    86.56   |   81.37   |    86.72   |
| SAMix*      |     2    |   81.11   |    86.83   |   82.15   |    86.80   |

### **Large-scale Fine-grained Datasets**

These benchmarks follow [PyTorch-style](https://arxiv.org/abs/2110.00476) ImageNet-1k training settings, training 100 epochs from stretch on [iNat2017](https://github.com/visipedia/inat_comp/tree/master/2017) and [iNat2018](https://github.com/visipedia/inat_comp/tree/master/2018).

**Note**

* Please refer to config files for experiment details: [iNat2017](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/inaturalist2017/) and [iNat2018](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/inaturalist2018/). As for config files of [various mixups](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/inaturalist2017/mixups/), please modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts.
* Download weights and logs: iNat2017 [[github](https://github.com/Westlake-AI/openmixup/releases/tag/mixup-inat2017-weights), [Baidu (1e7w)](https://pan.baidu.com/s/1GsoXVpIBXPjyFKsCdnmp9Q)], iNat2018 [[github](https://github.com/Westlake-AI/openmixup/releases/tag/mixup-inat2018-weights), [Baidu (wy2v)](https://pan.baidu.com/s/1P4VeJalFLV0chryjYCfveg)].

| Datasets    |  $Beta$  |  iNat2017 |  iNat2017 |   iNat2017  |  iNat2018 |   iNat2018  |
|-------------|:--------:|:---------:|:---------:|:-----------:|:---------:|:-----------:|
| Backbones   | $\alpha$ | ResNet-18 | ResNet-50 | ResNeXt-101 | ResNet-50 | ResNeXt-101 |
| Vanilla     |     -    |   51.79   |   60.23   |    63.70    |   62.53   |    66.94    |
| MixUp       |    0.2   |   51.40   |   61.22   |    66.27    |   62.69   |    67.56    |
| CutMix      |     1    |   51.24   |   62.34   |    67.59    |   63.91   |    69.75    |
| ManifoldMix |    0.2   |   51.83   |   61.47   |    66.08    |   63.46   |    69.30    |
| SaliencyMix |     1    |   51.29   |   62.51   |    67.20    |   64.27   |    70.01    |
| FMix*       |     1    |   52.01   |   61.90   |    66.64    |   63.71   |    69.46    |
| PuzzleMix   |     1    |     -     |   62.66   |    67.72    |   64.36   |    70.12    |
| ResizeMix*  |     1    |   51.21   |   62.29   |    66.82    |   64.12   |    69.30    |
| AutoMix     |     2    |   52.84   |   63.08   |    68.03    |   64.73   |    70.49    |
| SAMix*      |     2    |   53.42   |   63.32   |    68.26    |   64.84   |    70.54    |

### **Scenic Classification Dataset**

This benchmark follows [PyTorch-style](https://arxiv.org/abs/2110.00476) ImageNet-1k training settings, training 100 epochs from stretch on [Places205](http://places2.csail.mit.edu/index.html).

**Note**

* Please refer to config files of [Places205](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/place205/) for experiment details. As for config files of [various mixups](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/place205/mixups/), please modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts.
* Download weights and logs of Places205 [[github](https://github.com/Westlake-AI/openmixup/releases/tag/mixup-place205-weights), [Baidu (4m94)](https://pan.baidu.com/s/1ciAYxK6SwR13UNScp0W3bQ)].

| Datasets    |  $Beta$  |  Places205 |  Places205 |
|-------------|:--------:|:---------:|:---------:|
| Backbones   | $\alpha$ | ResNet-18 | ResNet-50 |
| Vanilla     |     -    |   59.63   |   63.10   |
| MixUp       |    0.2   |   59.33   |   63.01   |
| CutMix      |    0.2   |   59.21   |   63.75   |
| ManifoldMix |    0.2   |   59.46   |   63.23   |
| SaliencyMix |    0.2   |   59.50   |   63.33   |
| FMix*       |    0.2   |   59.51   |   63.63   |
| PuzzleMix   |     1    |   59.62   |   63.91   |
| ResizeMix*  |     1    |   59.66   |   63.88   |
| AutoMix     |     2    |   59.74   |   64.06   |
| SAMix*      |     2    |   59.86   |   64.27   |

<p align="right">(<a href="#top">back to top</a>)</p>