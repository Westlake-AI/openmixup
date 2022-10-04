# PVT

> [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)

## Abstract

Although using convolutional neural networks (CNNs) as backbones achieves great successes in computer vision, this work investigates a simple backbone network useful for many dense prediction tasks without convolutions. Unlike the recently-proposed Transformer model (e.g., ViT) that is specially designed for image classification, we propose Pyramid Vision Transformer~(PVT), which overcomes the difficulties of porting Transformer to various dense prediction tasks. PVT has several merits compared to prior arts. (1) Different from ViT that typically has low-resolution outputs and high computational and memory cost, PVT can be not only trained on dense partitions of the image to achieve high output resolution, which is important for dense predictions but also using a progressive shrinking pyramid to reduce computations of large feature maps. (2) PVT inherits the advantages from both CNN and Transformer, making it a unified backbone in various vision tasks without convolutions by simply replacing CNN backbones. (3) We validate PVT by conducting extensive experiments, showing that it boosts the performance of many downstream tasks, e.g., object detection, semantic, and instance segmentation. For example, with a comparable number of parameters, RetinaNet+PVT achieves 40.4 AP on the COCO dataset, surpassing RetinNet+ResNet50 (36.3 AP) by 4.1 absolute AP. We hope PVT could serve as an alternative and useful backbone for pixel-level predictions and facilitate future researches. Code is available at this https URL.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/193921646-69c04345-35e6-4c94-b5fb-813b8229de3e.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

|     Model    | resolution | Params(M) | Flops(G) | Top-1 (%) |                                                            Config                                                           |                                      Download                                     |
|:------------:|:----------:|:---------:|:--------:|:---------:|:---------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|
|  PVT-Tiny\*  |   224x224  |    13.2   |   1.60   |    75.1   |  [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/pvt/pvt_tiny_8xb128_ep300.py)  |  [model](https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth) / log  |
|  PVT-Small\* |   224x224  |    24.5   |   3.80   |    79.8   |  [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/pvt/pvt_small_8xb128_ep300.py) |  [model](https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth) / log |
| PVT-Medium\* |   224x224  |    44.2   |   6.70   |    81.2   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/pvt/pvt_medium_8xb128_ep300.py) |  [model](https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth) / log |
|  PVT-Large\* |   224x224  |    61.2   |   9.80   |    81.7   |  [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/pvt/pvt_large_8xb128_ep300.py) | [model](https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth)  / log |

We follow the original training setting provided by the original paper. *Models with * are converted from the [official repo](https://github.com/whai362/PVT).* We don't ensure these config files' training accuracy.

## Citation

```
@article{iccv2021PVT,
  title={Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions},
  author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
  journal={2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021},
  pages={548-558}
}
```
