# BEiT

> [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)

## Abstract

We introduce a self-supervised vision representation model BEiT, which stands for Bidirectional Encoder representation from Image Transformers. Following BERT developed in the natural language processing area, we propose a masked image modeling task to pretrain vision Transformers. Specifically, each image has two views in our pre-training, i.e, image patches (such as 16x16 pixels), and visual tokens (i.e., discrete tokens). We first "tokenize" the original image into visual tokens. Then we randomly mask some image patches and fed them into the backbone Transformer. The pre-training objective is to recover the original visual tokens based on the corrupted image patches. After pre-training BEiT, we directly fine-tune the model parameters on downstream tasks by appending task layers upon the pretrained encoder. Experimental results on image classification and semantic segmentation show that our model achieves competitive results with previous pre-training methods. For example, base-size BEiT achieves 83.2% top-1 accuracy on ImageNet-1K, significantly outperforming from-scratch DeiT training (81.8%) with the same setup. Moreover, large-size BEiT obtains 86.3% only using ImageNet-1K, even outperforming ViT-L with supervised pre-training on ImageNet-22K (85.2%).

<div align=center>
<img src="https://user-images.githubusercontent.com/36138628/203688351-adac7146-4e71-4ab6-8958-5cfe643a2dc5.png" width="95%"/>
</div>

## Models and results

### Image Classification on ImageNet-1k

|   Model    | Pretrain  | Params(M)  | Flops(G)  | Top-1 (%) | Top-5 (%) | Config | Download |
| :--------: | :-------: | :--------: | :-------: | :-------: | :-------: | :----: | :------: |
| BEiT-Base  |    BEiT   |   86.53    |   17.58   |   83.10   |    N/A    | [config](beit_base_8xb128_fp16_ep300.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/beit/beit_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221128-0ca393e9.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/beit/beit_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221128-0ca393e9.json) |

*Models with * are converted from the [official repo](https://github.com/microsoft/unilm/tree/master/beit). The config files of these models are only for inference. We haven't reprodcue the training results.*

## Citation

```bibtex
@inproceedings{bao2022beit,
    title={{BE}iT: {BERT} Pre-Training of Image Transformers},
    author={Hangbo Bao and Li Dong and Songhao Piao and Furu Wei},
    booktitle={International Conference on Learning Representations},
    year={2022},
}
```
