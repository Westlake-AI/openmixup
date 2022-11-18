# Efficient Multi-order Gated Aggregation Network

> [Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295)

## Abstract

Since the recent success of Vision Transformers (ViTs), explorations toward transformer-style architectures have triggered the resurgence of modern ConvNets. In this work, we explore the representation ability of DNNs through the lens of interaction complexities. We empirically show that interaction complexity is an overlooked but essential indi-cator for visual recognition. Accordingly, a new family of efﬁcient ConvNets, named MogaNet, is presented to pursue informative context mining in pure ConvNet-based models, with preferable complexity-performance trade-offs. In MogaNet, interactions across multiple complexities are facil-itated and contextualized by leveraging two specially designed aggregation blocks in both spatial and channel interaction spaces. Extensive studies are conducted on ImageNet classiﬁcation, COCO object detection, and ADE20K semantic segmentation tasks. The results demonstrate that our MogaNet establishes new state-of-the-art over other popular methods in mainstream scenarios and all model scales. Typically, the lightweight MogaNet-T achieves 80.0% top-1 accuracy with only 1.44G FLOPs using reﬁned training setup on ImageNet-1K, surpassing ParC-Net-S by 1.4% accuracy but saving 59% (2.04G) FLOPs.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/200625735-86bd2237-5bbe-43c1-ab37-049810b8d8a1.jpg" width="100%"/>
</div>

## Results and models

Here, we provide ImageNet classification results and pre-trained models. Please refer to [code](https://github.com/Westlake-AI/MogaNet) for full implementations of downstream tasks (COCO object detection and ADE20K semantic segmentation).

### ImageNet-1k

|     Model    |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) |                               Config                                |                               Download                                |
| :----------: | :----------: | :--------: | :-------: | :------: | :-------: | :-----------------------------------------------------------------: | :-------------------------------------------------------------------: |
|  MogaNet-XT  | From scratch |  224x224   |    2.97   |   0.80   |    76.5   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_xtiny_sz224_8xb128_fp16_ep300.py) | model / log |
|  MogaNet-T   | From scratch |  224x224   |    5.20   |   1.10   |    79.0   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_tiny_sz224_8xb128_fp16_ep300.py)  | model / log |
|  MogaNet-T   | From scratch |  256x256   |    5.20   |   1.44   |    79.6   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_tiny_sz256_8xb128_fp16_ep300.py)  | model / log |
|  MogaNet-T\* | From scratch |  256x256   |    5.20   |   1.44   |    80.0   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_tiny_deit3_sz256_lr2e_3_8xb128_fp16_ep300.py) | model / log |
|  MogaNet-S   | From scratch |  224x224   |    25.3   |   4.97   |    83.4   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_small_sz224_8xb128_ep300.py)      | model / log |
|  MogaNet-B   | From scratch |  224x224   |    43.9   |   9.93   |    84.2   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_base_sz224_8xb128_ep300.py)       | model / log |
|  MogaNet-L   | From scratch |  224x224   |    82.5   |   15.9   |    84.6   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_large_sz224_8xb64_accu2_ep300.py) | model / log |

We provide the config files according to the original training setting described in the [paper](https://arxiv.org/abs/2211.03295). Note that \* denotes the refined training setting of lightweight models and we can get a slight better performance with [Precise BN](https://arxiv.org/abs/2105.07576).

## Citation

```
@article{Li2022MogaNet,
  title={Efficient Multi-order Gated Aggregation Network},
  author={Siyuan Li and Zedong Wang and Zicheng Liu and Cheng Tan and Haitao Lin and Di Wu and Zhiyuan Chen and Jiangbin Zheng and Stan Z. Li},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.03295}
}
```
