# EdgeNeXt

> [EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](https://arxiv.org/abs/2206.10589)

## Abstract

In the pursuit of achieving ever-increasing accuracy, large and complex neural networks are usually developed. Such models demand high computational resources and therefore cannot be deployed on edge devices. It is of great interest to build resource-efficient general purpose networks due to their usefulness in several application areas. In this work, we strive to effectively combine the strengths of both CNN and Transformer models and propose a new efficient hybrid architecture EdgeNeXt. Specifically in EdgeNeXt, we introduce split depth-wise transpose attention (SDTA) encoder that splits input tensors into multiple channel groups and utilizes depth-wise convolution along with self-attention across channel dimensions to implicitly increase the receptive field and encode multi-scale features. Our extensive experiments on classification, detection and segmentation tasks, reveal the merits of the proposed approach, outperforming state-of-the-art methods with comparatively lower compute requirements. Our EdgeNeXt model with 1.3M parameters achieves 71.2% top-1 accuracy on ImageNet-1K, outperforming MobileViT with an absolute gain of 2.2% with 28% reduction in FLOPs. Further, our EdgeNeXt model with 5.6M parameters achieves 79.4% top-1 accuracy on ImageNet-1K.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/189606507-f765483c-d9d7-4a2e-8b32-c8d4d4642b85.png" width="95%"/>
</div>

## Results and models

### ImageNet-1k

|        Model         |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                              Config                               |                               Download                               |
| :------------------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------------------------------: | :------------------------------------------------------------------: |
|   EdgeNeXt-Base\*    | From scratch |  256x256  |   18.51   |   3.84   |   82.48   |   96.2    | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/edgenext/edgenext_base_sz256_8xb256_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/edgenext/edgenext-base_3rdparty_8xb256_in1k_20220801-9ade408b.pth) |
|   EdgeNeXt-Small\*   | From scratch |  256x256  |   5.59    |   1.26   |   79.41   |   94.53   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/edgenext/edgenext_base_sz256_8xb256_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/edgenext/edgenext-small_3rdparty_8xb256_in1k_20220801-d00db5f8.pth) |
|  EdgeNeXt-X-Small\*  | From scratch |  256x256  |   2.34    |  0.538   |   74.86   |   92.31   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/edgenext/edgenext_base_sz256_8xb256_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/edgenext/edgenext-xsmall_3rdparty_8xb256_in1k_20220801-974f9fe7.pth) |
| EdgeNeXt-XX-Small\*  | From scratch |  256x256  |   1.33    |  0.261   |   71.2    |   89.91   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/edgenext/edgenext_base_sz256_8xb256_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/edgenext/edgenext-xxsmall_3rdparty_8xb256_in1k_20220801-7ca8a81d.pth) |

We follow the original training setting provided by the [official repo](https://github.com/mmaaz60/EdgeNeXt) and the [original paper](https://arxiv.org/abs/2206.10589). *Note that models with \* are converted from the [official repo](https://github.com/mmaaz60/EdgeNeXt).*

## Citation

```
@article{Maaz2022EdgeNeXt,
    title={EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications},
    author={Muhammad Maaz and Abdelrahman Shaker and Hisham Cholakkal and Salman Khan and Syed Waqas Zamir and Rao Muhammad Anwer and Fahad Shahbaz Khan},
    journal={2206.10589},
    year={2022}
}
```
