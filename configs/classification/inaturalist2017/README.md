# Mixup Classification Benchmark on iNaturalist-2017

> [The iNaturalist Challenge 2017 Dataset](https://arxiv.org/abs/1707.06642)

## Abstract

Existing image classification datasets used in computer vision tend to have an even number of images for each object category. In contrast, the natural world is heavily imbalanced, as some species are more abundant and easier to photograph than others. To encourage further progress in challenging real world conditions we present the iNaturalist Challenge 2017 dataset - an image classification benchmark consisting of 675,000 images with over 5,000 different species of plants and animals. It features many visually similar species, captured in a wide variety of situations, from all over the world. Images were collected with different camera types, have varying image quality, have been verified by multiple citizen scientists, and feature a large class imbalance. We discuss the collection of the dataset and present baseline results for state-of-the-art computer vision classification models. Results show that current non-ensemble based methods achieve only 64% top one classification accuracy, illustrating the difficulty of the dataset. Finally, we report results from a competition that was held with the data.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/185645586-a5317b04-daf4-429c-a0fa-c9dd87dac45b.png" width="100%"/>
</div>

## Results and models

We provide a collection of [weights and logs](https://github.com/Westlake-AI/openmixup/releases/tag/mixup-inat2017-weights) for mixup classification benchmark on iNaturalist-2017. You can download all results from **Baidu Cloud**: [iNaturalist-2017 (1e7w)](https://pan.baidu.com/s/1GsoXVpIBXPjyFKsCdnmp9Q).

* All compared methods adopt ResNet-18/50 and ResNeXt-101 (32x4d) architectures and are trained 100 epochs using the PyTorch training recipe. The training and testing image size is 224 with the CenterCrop ratio of 0.85. We search $\alpha$ in $Beta(\alpha, \alpha)$ for all compared methods.
* Please refer to [configs]((https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/inaturalist2017/)) files for experiment details. You can modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts.
* The **median** of top-1 accuracy in the last 5 training epochs is reported for ResNet variants.
* Visualization of mixed samples from [AutoMix](https://arxiv.org/abs/2103.13027) and [SAMix](https://arxiv.org/abs/2111.15454) are provided in zip files.

### iNaturalist-2017

| Backbones                                                   | ResNet-18 top-1 | ResNet-50 top-1 | ResNeXt-101 top-1 |
|-------------------------------------------------------------|:---------------:|:---------------:|:-----------------:|
| Vanilla                                                     |      51.79      |      60.23      |       63.70       |
| MixUp [[ICLR'2018](https://arxiv.org/abs/1710.09412)]       |      51.40      |      61.22      |       66.27       |
| CutMix [[ICCV'2019](https://arxiv.org/abs/1905.04899)]      |      51.24      |      62.34      |       67.59       |
| ManifoldMix [[ICML'2019](https://arxiv.org/abs/1806.05236)] |      51.83      |      61.47      |       66.08       |
| SaliencyMix [[ICLR'2021](https://arxiv.org/abs/2006.01791)] |      51.29      |      62.51      |       67.20       |
| FMix [[Arixv'2020](https://arxiv.org/abs/2002.12047)]       |      52.01      |      61.90      |       66.64       |
| PuzzleMix [[ICML'2020](https://arxiv.org/abs/2009.06962)]   |        -        |      62.66      |       67.72       |
| ResizeMix [[Arixv'2020](https://arxiv.org/abs/2012.11101)]  |      51.21      |      62.29      |       66.82       |
| AutoMix [[ECCV'2022](https://arxiv.org/abs/2103.13027)]     |      52.84      |      63.08      |       68.03       |
| SAMix [[Arxiv'2021](https://arxiv.org/abs/2111.15454)]      |      53.42      |      63.32      |       68.26       |

We summarize mixup benchmarks in [Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md).


## Citation

Please refer to the original paper of [iNaturalist](https://arxiv.org/abs/1707.06642) for dataset information, and refer to [AutoMix](https://arxiv.org/abs/2103.13027) for experiment details.

```bibtex
@article{Horn2017TheIC,
  title={The iNaturalist Challenge 2017 Dataset},
  author={Grant Van Horn and Oisin Mac Aodha and Yang Song and Alexander Shepard and Hartwig Adam and Pietro Perona and Serge J. Belongie},
  journal={ArXiv},
  year={2017},
  volume={abs/1707.06642}
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
