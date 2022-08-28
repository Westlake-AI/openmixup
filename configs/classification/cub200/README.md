# Mixup Classification Benchmark on CUB-200-2011

> [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)

## Abstract

The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset is the most widely-used dataset for fine-grained visual categorization task. It contains 11,788 images of 200 subcategories belonging to birds, 5,994 for training and 5,794 for testing. Each image has detailed annotations: 1 subcategory label, 15 part locations, 312 binary attributes and 1 bounding box. The textual information comes from Reed et al.. They expand the CUB-200-2011 dataset by collecting fine-grained natural language descriptions. Ten single-sentence descriptions are collected for each image. The natural language descriptions are collected through the Amazon Mechanical Turk (AMT) platform, and are required at least 10 words, without any information of subcategories and actions.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/185861496-c452d8a4-ac18-411f-a49b-f2c50c6770c1.jpeg" width="100%"/>
</div>

## Results and models

* This benchmark follows transfer learning settings on fine-grained datasets, using PyTorch official pre-trained models as initialization and training ResNet-18 and ResNeXt-50 (32x4d) architectures for 200 epochs on [CUB-200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). The training and testing image size is 224 with the RandomResizedCrop ratio of 0.5 and the CenterCrop ratio of 0.85. We search $\alpha$ in $Beta(\alpha, \alpha)$ for all compared methods.
* Please refer to [configs](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/cub200/mixups/basic/) for experiment details. You can modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts.
* The **median** of top-1 accuracy in the last 10 training epochs is reported for ResNet-18 and ResNeXt-50.

### CUB-200-2011

| Backbones                                                   | ResNet-18 top-1 | ResNeXt-50 top-1 |
|-------------------------------------------------------------|:---------------:|:----------------:|
| Vanilla                                                     |      77.68      |       83.01      |
| MixUp [[ICLR'2018](https://arxiv.org/abs/1710.09412)]       |      78.39      |       84.58      |
| CutMix [[ICCV'2019](https://arxiv.org/abs/1905.04899)]      |      78.40      |       85.68      |
| ManifoldMix [[ICML'2019](https://arxiv.org/abs/1806.05236)] |      79.76      |       86.38      |
| SaliencyMix [[ICLR'2021](https://arxiv.org/abs/2006.01791)] |      77.95      |       83.29      |
| FMix [[Arixv'2020](https://arxiv.org/abs/2002.12047)]       |      77.28      |       84.06      |
| PuzzleMix [[ICML'2020](https://arxiv.org/abs/2009.06962)]   |      78.63      |       84.51      |
| ResizeMix [[Arixv'2020](https://arxiv.org/abs/2012.11101)]  |      78.50      |       84.77      |
| AutoMix [[ECCV'2022](https://arxiv.org/abs/2103.13027)]     |      79.87      |       86.56      |
| SAMix [[Arxiv'2021](https://arxiv.org/abs/2111.15454)]      |      81.11      |       86.83      |

We summarize mixup benchmarks in [Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md).


## Citation

Please refer to the original paper of [CUB-200](http://www.vision.caltech.edu/datasets/cub_200_2011/) and [AutoMix](https://arxiv.org/abs/2103.13027) for details.

```bibtex
@techreport{2011CUB,
	Title = ,
	Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
	Year = {2011}
	Institution = {California Institute of Technology},
	Number = {CNS-TR-2011-001}
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
