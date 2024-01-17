# AdAutomixup 

> [Adversarial AutoMixup](https://arxiv.org/abs/2312.11954)

## Abstract

Data mixing augmentation has been widely applied to improve the generalization ability of deep neural networks. Recently, offline data mixing augmentation, e.g. handcrafted and saliency information-based mixup, has been gradually replaced by automatic mixing approaches. Through minimizing two sub-tasks, namely, mixed sample generation and mixup classification in an end-to-end way, AutoMix significantly improves accuracy on image classification tasks. However, as the optimization objective is consistent for the two sub-tasks, this approach is prone to generating consistent instead of diverse mixed samples, which results in overfitting for target task training. In this paper, we propose AdAutomixup, an adversarial automatic mixup augmentation approach that generates challenging samples to train a robust classifier for image classification, by alternatively optimizing the classifier and the mixup sample generator. AdAutomixup comprises two modules, a mixed example generator, and a target classifier. The mixed sample generator aims to produce hard mixed examples to challenge the target classifier while the target classifier`s aim is to learn robust features from hard mixed examples to improve generalization. To prevent the collapse of the inherent meanings of images, we further introduce an exponential moving average (EMA) teacher and cosine similarity to train AdAutomixup in an end-to-end way. Extensive experiments on seven image benchmarks consistently prove that our approach outperforms the state of the art in various classification scenarios.

<div align=center>
<img src="https://github.com/JinXins/Adversarial-AutoMixup/assets/124172716/c8b2f194-41b1-4117-8965-68c9c20d3c83" width="80%"/>
</div>

## Results and models

### ImageNet-1k

|    Model    |  Mixup  | resolution | Params(M) | Epochs | Top-1 (%) |                        Config                             |   Download  |
|:-----------:|:-------:|:----------:|:---------:|:------:|:---------:|:---------------------------------------------------------:|:-----------:|
|  ResNet-18  | AutoMix |   224x224  |   11.17   |   100  |   70.86   |        [config](./basic/r18_adautomix_bilinear.py)        | [model](https://github.com/JinXins/Adversarial-AutoMixup/releases/download/imagenet/ImageNet1k_adautomix_r18_100e.pth) |
|  ResNet-50  | AutoMix |   224x224  |   21.28   |   100  |   74.82   |        [config](./basic/r50_adautomix_bilinear.py)        | [model](https://github.com/JinXins/Adversarial-AutoMixup/releases/download/imagenet/ImageNet1k_adautomix_r34_100e.pth) |
|  ResNet-50  | AutoMix |   224x224  |   23.52   |   100  |   78.04   |        [config](./basic/r50_adautomix_bilinear.py)        | [model](https://github.com/JinXins/Adversarial-AutoMixup/releases/download/imagenet/ImageNet1k_adautomix_r50_100e.pth) |

Refer to [official repo](https://github.com/jinxins/adversarial-automixup) for configs and models. Please refer to [Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md) for image classification results, and refer to [Awesome-Mixup](https://github.com/Westlake-AI/Awesome-Mixup) for more Mixup methods.

## Find-Grained Datasets

| Name             | alpha | Confrence  | CUB R18 | CUB R50 | CUB RX50 | FGVC R18 | FGVC RX50 | Cars R18 | Cars RX50 |
|------------------|-------|------------|---------|---------|----------|--------------------|---------------------|--------------------|---------------------|
| AutoMix          | 2.0   | ECCV'2022  | 79.87   | 83.88   | 86.56    | 81.37              | 86.72               | 88.89              | 91.38               |
| SAMix            | 2.0   | ArXiv'2022 | 81.11   | 84.10   | 86.33    | 82.15              | 86.80               | 89.14              | 90.46               |
| AdAutoMix        | 1.0   | ICLR'2024  | **80.88**  | **84.57**   | - | **81.73**          | **87.16**           | **89.19**          | **91.59**           |

## Citation

```bibtex
@inproceedings{iclr2024adautomix,
      title={Adversarial AutoMixup},
      author={Huafeng Qin and Xin Jin and Yun Jiang and Mounim A. El-Yacoubi and Xinbo Gao},
      booktitle={International Conference on Learning Representations},
      year={2024},
}
```
