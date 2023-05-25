# MobileViT

> [MobileViT Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)

## Introduction

**MobileViT** aims at introducing a light-weight network, which takes the advantages of both ViTs and CNNs, uses the `InvertedResidual` blocks in [MobileNetV2](../mobilenet_v2/README.md) and `MobileViTBlock` which refers to [ViT](../vision_transformer/README.md) transformer blocks to build a standard 5-stage model structure.

The MobileViTBlock reckons transformers as convolutions to perform a global representation, meanwhile conbined with original convolution layers for local representation to build a block with global receptive field. This is different from ViT, which adds an extra class token and position embeddings for learning relative relationship. Without any position embeddings, MobileViT can benfit from multi-scale inputs during training.

Also, this paper puts forward a strategy for multi-scale training to dynamically adjust batch size based on the image size to both improve training efficiency and final performance.

It is also proven effective in downstream tasks such as object detection and segmentation.

<div align=center>
<img src="https://user-images.githubusercontent.com/42952108/193229983-822bf025-89a6-4d95-b6be-76b7f1a62f2c.png" width="70%"/>
</div>

## Abstract

<details>

<summary>Show the paper's abstract</summary>

<br>

Light-weight convolutional neural networks (CNNs) are the de-facto for mobile vision tasks. Their spatial inductive biases allow them to learn representations with fewer parameters across different vision tasks. However, these networks are spatially local. To learn global representations, self-attention-based vision trans-formers (ViTs) have been adopted. Unlike CNNs, ViTs are heavy-weight. In this paper, we ask the following question: is it possible to combine the strengths of CNNs and ViTs to build a light-weight and low latency network for mobile vision tasks? Towards this end, we introduce MobileViT, a light-weight and general-purpose vision transformer for mobile devices. MobileViT presents a different perspective for the global processing of information with transformers, i.e., transformers as convolutions. Our results show that MobileViT significantly outperforms CNN- and ViT-based networks across different tasks and datasets. On the ImageNet-1k dataset, MobileViT achieves top-1 accuracy of 78.4% with about 6 million parameters, which is 3.2% and 6.2% more accurate than MobileNetv3 (CNN-based) and DeIT (ViT-based) for a similar number of parameters. On the MS-COCO object detection task, MobileViT is 5.7% more accurate than MobileNetv3 for a similar number of parameters.
</br>

</details>

## Models and results

### Image Classification on ImageNet-1k

| Model                               |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                   Config                   |                                  Download                                  |
| :---------------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :----------------------------------------: | :------------------------------------------------------------------------: |
| `mobilevit-small_3rdparty_in1k`\*   | From scratch |    5.58    |   2.03    |   78.25   |   94.09   |  [config](mobilevit_small_8xb128_ep300.py)  | [model](https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth) |
| `mobilevit-xsmall_3rdparty_in1k`\*  | From scratch |    2.32    |   1.05    |   74.75   |   92.32   | [config](mobilevit_xsmall_8xb128_ep300.py)  | [model](https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-xsmall_3rdparty_in1k_20221018-be39a6e7.pth) |
| `mobilevit-xxsmall_3rdparty_in1k`\* | From scratch |    1.27    |   0.42    |   69.02   |   88.91   | [config](mobilevit_xxsmall_8xb128_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-xxsmall_3rdparty_in1k_20221018-77835605.pth) |

*Models with * are converted from the [official repo](https://github.com/apple/ml-cvnets). The config files of these models are only for inference. We haven't reprodcue the training results.*

## Citation

```bibtex
@inproceedings{mehta2022mobilevit,
  title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022},
}
```
