# MetaFormer Baselines for Vision

> [MetaFormer Baselines for Vision](https://arxiv.org/abs/2210.13452)

## Abstract

MetaFormer, the abstracted architecture of Transformer, has been found to play a significant role in achieving competitive performance. In this paper, we further explore the capacity of MetaFormer, again, without focusing on token mixer design: we introduce several baseline models under MetaFormer using the most basic or common mixers, and summarize our observations as follows: (1) MetaFormer ensures solid lower bound of performance. By merely adopting identity mapping as the token mixer, the MetaFormer model, termed IdentityFormer, achieves >80% accuracy on ImageNet-1K. (2) MetaFormer works well with arbitrary token mixers. When specifying the token mixer as even a random matrix to mix tokens, the resulting model RandFormer yields an accuracy of >81%, outperforming IdentityFormer. Rest assured of MetaFormer's results when new token mixers are adopted. (3) MetaFormer effortlessly offers state-of-the-art results. With just conventional token mixers dated back five years ago, the models instantiated from MetaFormer already beat state of the art. (a) ConvFormer outperforms ConvNeXt. Taking the common depthwise separable convolutions as the token mixer, the model termed ConvFormer, which can be regarded as pure CNNs, outperforms the strong CNN model ConvNeXt. (b) CAFormer sets new record on ImageNet-1K. By simply applying depthwise separable convolutions as token mixer in the bottom stages and vanilla self-attention in the top stages, the resulting model CAFormer sets a new record on ImageNet-1K: it achieves an accuracy of 85.5% at 224x224 resolution, under normal supervised training without external data or distillation. In our expedition to probe MetaFormer, we also find that a new activation, StarReLU, reduces 71% FLOPs of activation compared with GELU yet achieves better performance. We expect StarReLU to find great potential in MetaFormer-like models alongside other neural networks.  

<div align=center>
<img src="https://user-images.githubusercontent.com/49296856/212324452-ee5ccbcf-5577-42cb-9fa4-b9e6bdbb6d4a.png" width="99%"/>
</div>

## Results and models

This page is based on the [official repo](https://github.com/sail-sg/metaformer).

### ImageNet-1k

| Model | Resolution | Params | MACs | Top1 Acc | Download |
| :---     |   :---:    |  :---: |  :---:  |  :---:  |  :---:  |
| caformer_s18 | 224 | 26M | 4.1G |  83.6 | [here](https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth) |
| caformer_s18_384 | 384 | 26M | 13.4G |  85.0 | [here](https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_384.pth) |
| caformer_s36 | 224 | 39M | 8.0G |  84.5 | [here](https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36.pth) |
| caformer_s36_384 | 384 | 39M | 26.0G |  85.7 | [here](https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_384.pth) |
| caformer_m36 | 224 | 56M | 13.2G |  85.2 | [here](https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36.pth) |
| caformer_m36_384 | 384 | 56M | 42.0G |  86.2 | [here](https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384.pth) |
| caformer_b36 | 224 | 99M | 23.2G |  **85.5**\* | [here](https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36.pth) |
| caformer_b36_384 | 384 | 99M | 72.2G |  **86.4** | [here](https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384.pth) |
| convformer_s18 | 224 | 27M | 3.9G |  83.0 | [here](https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18.pth) |
| convformer_s18_384 | 384 | 27M | 11.6G |  84.4 | [here](https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_384.pth) |
| convformer_s36 | 224 | 40M | 7.6G |  84.1 | [here](https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36.pth) |
| convformer_s36_384 | 384 | 40M | 22.4G |  85.4 | [here](https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_384.pth) |
| convformer_m36 | 224 | 57M | 12.8G |  84.5 | [here](https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36.pth) |
| convformer_m36_384 | 384 | 57M | 37.7G |  85.6 | [here](https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_384.pth) |
| convformer_b36 | 224 | 100M | 22.6G |  84.8 | [here](https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36.pth) |
| convformer_b36_384 | 384 | 100M | 66.5G |  85.7 | [here](https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_384.pth) |

We mainly follow the original training setting provided by the [official repo](https://github.com/sail-sg/metaformer) to construct config files. *Models with * are converted from the [official repo](https://github.com/sail-sg/metaformer).*

## Citation

```bibtex
@article{yu2022metaformer,
  title={Metaformer baselines for vision},
  author={Yu, Weihao and Si, Chenyang and Zhou, Pan and Luo, Mi and Zhou, Yichen and Feng, Jiashi and Yan, Shuicheng and Wang, Xinchao},
  journal={arXiv preprint arXiv:2210.13452},
  year={2022}
}
```
