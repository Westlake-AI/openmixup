# BEiT.V1

> [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)

## Abstract

We introduce a self-supervised vision representation model BEiT, which stands for Bidirectional Encoder representation from Image Transformers. Following BERT developed in the natural language processing area, we propose a masked image modeling task to pretrain vision Transformers. Specifically, each image has two views in our pre-training, i.e, image patches (such as 16x16 pixels), and visual tokens (i.e., discrete tokens). We first "tokenize" the original image into visual tokens. Then we randomly mask some image patches and fed them into the backbone Transformer. The pre-training objective is to recover the original visual tokens based on the corrupted image patches. After pre-training BEiT, we directly fine-tune the model parameters on downstream tasks by appending task layers upon the pretrained encoder. Experimental results on image classification and semantic segmentation show that our model achieves competitive results with previous pre-training methods. For example, base-size BEiT achieves 83.2% top-1 accuracy on ImageNet-1K, significantly outperforming from-scratch DeiT training (81.8%) with the same setup. Moreover, large-size BEiT obtains 86.3% only using ImageNet-1K, even outperforming ViT-L with supervised pre-training on ImageNet-22K (85.2%). The code and pretrained models are available at this https URL. 

<div align="center">
<img src="https://user-images.githubusercontent.com/44519745/195199387-521de7cb-8989-4ed2-8dc0-0f79ca11ba91.png" width="90%"/>
</div>

## Models and Benchmarks

Here, we report the results provided in the [original repo](https://github.com/microsoft/unilm/tree/master/beit), which is pre-trained 800-epoch with ViT-Base on ImageNet-1k. To run the pre-training, please create a new folder `work_dirs/my_pretrains/beit_ckpt` under the root directory and download the
[pretrained weights](https://download.openmmlab.com/mmselfsup/cae/dalle_encoder.pth) for `dalle` encoder to the folder.

| Backbone | Pre-train epoch | Fine-tuning Top-1 |                                                            Pre-train Config                                                            |                                                                        Fine-tuning Config                                                                        |   Download   |
|:--------:|:---------------:|:-----------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------:|
| ViT-Base |       300       |        83.2       | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/beit/imagenet/vit_base_sz224_8xb64_accu4_cos_fp16_ep800.py) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_base_p16_swin_ft_simmim_sz224_4xb128_accu2_cos_ep100.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/beit/beit_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221128-0ca393e9.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/beit/beit_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221127_162126.json) |

## Citation

```bibtex
@article{iclr2022BEiT,
  title={BEiT: BERT Pre-Training of Image Transformers},
  author={Hangbo Bao and Li Dong and Furu Wei},
  journal={ArXiv},
  year={2022},
  volume={abs/2106.08254}
}
```
