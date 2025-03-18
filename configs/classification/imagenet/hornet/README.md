# HorNet

> [HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions](https://arxiv.org/abs/2207.14284)

## Abstract

Recent progress in vision Transformers exhibits great success in various tasks driven by the new spatial modeling mechanism based on dot-product self-attention. In this paper, we show that the key ingredients behind the vision Transformers, namely input-adaptive, long-range and high-order spatial interactions, can also be efficiently implemented with a convolution-based framework. We present the Recursive Gated Convolution (g nConv) that performs high-order spatial interactions with gated convolutions and recursive designs. The new operation is highly flexible and customizable, which is compatible with various variants of convolution and extends the two-order interactions in self-attention to arbitrary orders without introducing significant extra computation. g nConv can serve as a plug-and-play module to improve various vision Transformers and convolution-based models. Based on the operation, we construct a new family of generic vision backbones named HorNet. Extensive experiments on ImageNet classification, COCO object detection and ADE20K semantic segmentation show HorNet outperform Swin Transformers and ConvNeXt by a significant margin with similar overall architecture and training configurations. HorNet also shows favorable scalability to more training data and a larger model size. Apart from the effectiveness in visual encoders, we also show g nConv can be applied to task-specific decoders and consistently improve dense prediction performance with less computation. Our results demonstrate that g nConv can be a new basic module for visual modeling that effectively combines the merits of both vision Transformers and CNNs. Code is available at https://github.com/raoyongming/HorNet.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/188356236-b8e3db94-eaa6-48e9-b323-15e5ba7f2991.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

|     Model     |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                              Config                              |                              Download                              |
| :-----------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :--------------------------------------------------------------: | :----------------------------------------------------------------: |
|  HorNet-T\*   | From scratch |  224x224   |   22.41   |   3.98   |   82.84   |   96.24   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/hornet/hornet_tiny_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/hornet/hornet-tiny_3rdparty_in1k_20220915-0e8eedff.pth) |
| HorNet-T-GF\* | From scratch |  224x224   |   22.99   |   3.9    |   82.98   |   96.38   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/hornet/hornet_tiny_gf_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/hornet/hornet-tiny-gf_3rdparty_in1k_20220915-4c35a66b.pth) |
|  HorNet-S\*   | From scratch |  224x224   |   49.53   |   8.83   |   83.79   |   96.75   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/hornet/hornet_small_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/hornet/hornet-small_3rdparty_in1k_20220915-5935f60f.pth) |
| HorNet-S-GF\* | From scratch |  224x224   |   50.4    |   8.71   |   83.98   |   96.77   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/hornet/hornet_small_gf_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/hornet/hornet-small-gf_3rdparty_in1k_20220915-649ca492.pth) |
|  HorNet-B\*   | From scratch |  224x224   |   87.26   |  15.59   |   84.24   |   96.94   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/hornet/hornet_base_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/hornet/hornet-base_3rdparty_in1k_20220915-a06176bb.pth) |
| HorNet-B-GF\* | From scratch |  224x224   |   88.42   |  15.42   |   84.32   |   96.95   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/hornet/hornet_base_gf_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/hornet/hornet-base-gf_3rdparty_in1k_20220915-82c06fa7.pth) |

We follow the original training setting provided by the [official repo](https://github.com/raoyongming/HorNet) and the [original paper](https://arxiv.org/abs/2207.14284). *Note that models with \* are converted from the [official repo](https://github.com/raoyongming/HorNet).*

## Citation

```
<<<<<<< HEAD
@article{rao2022hornet,
  title={HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions},
  author={Rao, Yongming and Zhao, Wenliang and Tang, Yansong and Zhou, Jie and Lim, Ser-Lam and Lu, Jiwen},
  journal={arXiv preprint arXiv:2207.14284},
  year={2022}
=======
@inproceedings{nips2022hornet,
    title={HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions},
    author={Yongming Rao and Wenliang Zhao and Yansong Tang and Jie Zhou and Ser Nam Lim and Jiwen Lu},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2022},
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
}
```
