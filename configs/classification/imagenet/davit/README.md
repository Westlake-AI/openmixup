# DaViT

> [DaViT: Dual Attention Vision Transformers](https://arxiv.org/abs/2204.03645)

## Abstract

In this work, we introduce Dual Attention Vision Transformers (DaViT), a simple yet effective vision transformer architecture that is able to capture global context while maintaining computational efficiency. We propose approaching the problem from an orthogonal angle: exploiting self-attention mechanisms with both "spatial tokens" and "channel tokens". With spatial tokens, the spatial dimension defines the token scope, and the channel dimension defines the token feature dimension. With channel tokens, we have the inverse: the channel dimension defines the token scope, and the spatial dimension defines the token feature dimension. We further group tokens along the sequence direction for both spatial and channel tokens to maintain the linear complexity of the entire model. We show that these two self-attentions complement each other: (i) since each channel token contains an abstract representation of the entire image, the channel attention naturally captures global interactions and representations by taking all spatial positions into account when computing attention scores between channels; (ii) the spatial attention refines the local representations by performing fine-grained interactions across spatial locations, which in turn helps the global information modeling in channel attention. Extensive experiments show our DaViT achieves state-of-the-art performance on four different tasks with efficient computations. Without extra data, DaViT-Tiny, DaViT-Small, and DaViT-Base achieve 82.8%, 84.2%, and 84.6% top-1 accuracy on ImageNet-1K with 28.3M, 49.7M, and 87.9M parameters, respectively. When we further scale up DaViT with 1.5B weakly supervised image and text pairs, DaViT-Gaint reaches 90.4% top-1 accuracy on ImageNet-1K.

<div align=center>
<img src="https://user-images.githubusercontent.com/24734142/196125065-e232409b-f710-4729-b657-4e5f9158f2d1.png" width="90%"/>
</div>

## Models and results

### Image Classification on ImageNet-1k

| Model                         |   Pretrain   | Params (M) | Flops (G) | Top-1 (%) | Top-5 (%) |                Config                |                                        Download                                        |
| :---------------------------- | :----------: | :--------: | :-------: | :-------: | :-------: | :----------------------------------: | :------------------------------------------------------------------------------------: |
| `davit-tiny_3rdparty_in1k`\*  | From scratch |   28.36    |   4.54    |   82.24   |   96.13   | [config](davit_tiny_sz224_4xb256_ep300.py)  | [model](https://download.openmmlab.com/mmclassification/v0/davit/davit-tiny_3rdparty_in1k_20221116-700fdf7d.pth) |
| `davit-small_3rdparty_in1k`\* | From scratch |   49.75    |   8.80    |   83.61   |   96.75   | [config](davit_small_sz224_4xb256_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/davit/davit-small_3rdparty_in1k_20221116-51a849a6.pth) |
| `davit-base_3rdparty_in1k`\*  | From scratch |   87.95    |   15.51   |   84.09   |   96.82   | [config](davit_base_sz224_4xb256_ep300.py)  | [model](https://download.openmmlab.com/mmclassification/v0/davit/davit-base_3rdparty_in1k_20221116-19e0d956.pth) |

*Models with * are converted from the [official repo](https://github.com/dingmyu/davit/blob/main/mmdet/mmdet/models/backbones/davit.py#L355). The config files of these models are only for inference. We haven't reprodcue the training results.*

## Citation

```bibtex
@inproceedings{ding2022davit,
    title={DaViT: Dual Attention Vision Transformer},
    author={Ding, Mingyu and Xiao, Bin and Codella, Noel and Luo, Ping and Wang, Jingdong and Yuan, Lu},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022},
}
```
