# Mixup Classification Benchmark on FGVC-Aircraft

> [Fine-Grained Visual Classification of Aircraft](https://arxiv.org/abs/1306.5151)

## Abstract

This paper introduces FGVC-Aircraft, a new dataset containing 10,000 images of aircraft spanning 100 aircraft models, organised in a three-level hierarchy. At the finer level, differences between models are often subtle but always visually measurable, making visual recognition challenging but possible. A benchmark is obtained by defining corresponding classification tasks and evaluation protocols, and baseline results are presented. The construction of this dataset was made possible by the work of aircraft enthusiasts, a strategy that can extend to the study of number of other object classes. Compared to the domains usually considered in fine-grained visual classification (FGVC), for example animals, aircraft are rigid and hence less deformable. They, however, present other interesting modes of variation, including purpose, size, designation, structure, historical style, and branding. 

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/185860274-c7501d34-4d3c-438f-a492-966d3b27cbbc.jpg" width="100%"/>
</div>

## Results and models

* This benchmark follows transfer learning settings on fine-grained datasets, using PyTorch official pre-trained models as initialization and training ResNet-18 and ResNeXt-50 (32x4d) architectures for 200 epochs on [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/). The training and testing image size is 224 with the RandomResizedCrop ratio of 0.5 and the CenterCrop ratio of 0.85. We search $\alpha$ in $Beta(\alpha, \alpha)$ for all compared methods.
* Please refer to [configs](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/aircrafts/mixups/basic) for experiment details. You can modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts.
* The **median** of top-1 accuracy in the last 10 training epochs is reported for ResNet-18 and ResNeXt-50.

### FGVC-Aircraft

| Backbones                                                   | ResNet-18 top-1 | ResNeXt-50 top-1 |
|-------------------------------------------------------------|:---------------:|:----------------:|
| Vanilla                                                     |      80.23      |       85.10      |
| MixUp [[ICLR'2018](https://arxiv.org/abs/1710.09412)]       |      79.52      |       85.18      |
| CutMix [[ICCV'2019](https://arxiv.org/abs/1905.04899)]      |      78.84      |       84.55      |
| ManifoldMix [[ICML'2019](https://arxiv.org/abs/1806.05236)] |      80.68      |       86.60      |
| SaliencyMix [[ICLR'2021](https://arxiv.org/abs/2006.01791)] |      80.02      |       84.31      |
| FMix [[Arixv'2020](https://arxiv.org/abs/2002.12047)]       |      79.36      |       86.23      |
| PuzzleMix [[ICML'2020](https://arxiv.org/abs/2009.06962)]   |      80.76      |       86.23      |
| ResizeMix [[Arixv'2020](https://arxiv.org/abs/2012.11101)]  |      78.10      |       84.08      |
| AutoMix [[ECCV'2022](https://arxiv.org/abs/2103.13027)]     |      81.37      |       86.72      |
| SAMix [[Arxiv'2021](https://arxiv.org/abs/2111.15454)]      |      82.15      |       86.80      |

We summarize mixup benchmarks in [Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md).


## Citation

Please refer to the original paper of [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) and [AutoMix](https://arxiv.org/abs/2103.13027) for details.

```bibtex
@techreport{maji13fine-grained,
   title         = {Fine-Grained Visual Classification of Aircraft},
   author        = {S. Maji and J. Kannala and E. Rahtu
                    and M. Blaschko and A. Vedaldi},
   year          = {2013},
   archivePrefix = {arXiv},
   eprint        = {1306.5151},
   primaryClass  = "cs-cv",
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
