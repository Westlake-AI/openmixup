# MViT V2

> [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://arxiv.org/abs/2112.01526)

## Abstract

In this paper, we study Multiscale Vision Transformers (MViTv2) as a unified architecture for image and video classification, as well as object detection. We present an improved version of MViT that incorporates decomposed relative positional embeddings and residual pooling connections. We instantiate this architecture in five sizes and evaluate it for ImageNet classification, COCO detection and Kinetics video recognition where it outperforms prior work. We further compare MViTv2s' pooling attention to window attention mechanisms where it outperforms the latter in accuracy/compute. Without bells-and-whistles, MViTv2 has state-of-the-art performance in 3 domains: 88.8% accuracy on ImageNet classification, 58.7 boxAP on COCO object detection as well as 86.1% on Kinetics-400 video classification.


<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/180376227-755243fa-158e-4068-940a-416036519665.png" width="60%"/>
</div>

## Results and models

### ImageNet-1k

|     Model      |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                Config                                |                                Download                                 |
| :------------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :------------------------------------------------------------------: | :---------------------------------------------------------------------: |
| MViTv2-tiny\*  | From scratch |  224x224   |   24.17   |   4.70   |   82.33   |   96.15   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mvit/mvit_v2_tiny_8xb256_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-tiny_3rdparty_in1k_20220722-db7beeef.pth) |
| MViTv2-small\* | From scratch |  224x224   |   34.87   |   7.00   |   83.63   |   96.51   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mvit/mvit_v2_small_8xb256_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-small_3rdparty_in1k_20220722-986bd741.pth) |
| MViTv2-base\*  | From scratch |  224x224   |   51.47   |  10.20   |   84.34   |   96.86   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mvit/mvit_v2_base_8xb256_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-base_3rdparty_in1k_20220722-9c4f0a17.pth) |
| MViTv2-large\* | From scratch |  224x224   |  217.99   |  42.10   |   85.25   |   97.14   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/mvit/mvit_v2_large_8xb256_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-large_3rdparty_in1k_20220722-2b57b983.pth) |

We follow the original training setting provided by the [official repo](https://github.com/facebookresearch/mvit) and the [original paper](https://arxiv.org/abs/2112.01526). *Note that models with \* are converted from the [official repo](https://github.com/facebookresearch/mvit).*

## Citation

```
@inproceedings{li2021improved,
  title={MViTv2: Improved multiscale vision transformers for classification and detection},
  author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
