# SimMIM

> [SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886)

## Abstract

This paper presents SimMIM, a simple framework for masked image modeling. We simplify recently proposed related approaches without special designs such as blockwise masking and tokenization via discrete VAE or clustering. To study what let the masked image modeling task learn good representations, we systematically study the major components in our framework, and find that simple designs of each component have revealed very strong representation learning performance: 1) random masking of the input image with a moderately large masked patch size (e.g., 32) makes a strong pre-text task; 2) predicting raw pixels of RGB values by direct regression performs no worse than the patch classification approaches with complex designs; 3) the prediction head can be as light as a linear layer, with no worse performance than heavier ones. Using ViT-B, our approach achieves 83.8% top-1 fine-tuning accuracy on ImageNet-1K by pre-training also on this dataset, surpassing previous best approach by +0.6%. When applied on a larger model of about 650 million parameters, SwinV2H, it achieves 87.1% top-1 accuracy on ImageNet-1K using only ImageNet-1K data. We also leverage this approach to facilitate the training of a 3B model (SwinV2-G), that by 40× less data than that in previous practice, we achieve the state-of-the-art on four representative vision benchmarks. The code and models will be publicly available at https: //github.com/microsoft/SimMIM .

<div align="center">
<img src="https://user-images.githubusercontent.com/30762564/159404597-ac6d3a44-ee59-4cdc-8f6f-506a7d1b18b6.png" width="40%"/>
</div>

## Models and Benchmarks

Here, we report the results provided in the [original repo](https://github.com/microsoft/simmim), which is pre-trained 300/800-epoch on ImageNet-1k. More results will be coming soon.

| Backbone  | Pre-train epoch | Pre-train resolution | Fine-tuning resolution | Fine-tuning Top-1 |                                                           Pre-train Config                                                           |                                                                 Fine-tuning Config                                                                  |                                                                                                                           Download                                                                                                                            |
| :-------: | :-------------: | :------------------: | :--------------------: | :---------------: | :----------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ViT-Base  |       800       |         224          |          224           |       83.7        | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/vit_base_sz224_8xb128_accu2_step_fp16_ep800.py) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_base_p16_swin_ft_simmim_sz224_4xb128_accu2_cos_ep100.py) | model \| log |
| Swin-Base |       100       |         192          |          192           |       82.9        | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/swin_base_sz192_8xb128_accu2_cos_ep100.py)      | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/swin_base_swin_ft_sz192_4xb128_accu4_cos_ep100.py)   | [model](https://download.openmmlab.com/mmselfsup/simmim/simmim_swin-base_16xb128-coslr-100e_in1k-192_20220316-1d090125.pth) \| [log](https://download.openmmlab.com/mmselfsup/simmim/simmim_swin-base_16xb128-coslr-100e_in1k-192_20220316-1d090125.log.json) |
| Swin-Base |       100       |         192          |          224           |       83.5        | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/swin_base_sz192_8xb128_accu2_cos_ep100.py)      | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/swin_base_swin_ft_sz224_4xb128_accu4_cos_ep100.py) | [model](https://download.openmmlab.com/mmselfsup/simmim/simmim_swin-base_16xb128-coslr-100e_in1k-192_20220316-1d090125.pth) \| [log](https://download.openmmlab.com/mmselfsup/simmim/simmim_swin-base_16xb128-coslr-100e_in1k-192_20220316-1d090125.log.json) |
| Swin-Base |       800       |         192          |          224           |       84.0        | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/swin_base_sz192_8xb128_accu2_cos_ep800.py)      | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/swin_base_swin_ft_sz224_4xb128_accu4_cos_ep100.py) | model \| log |

## Citation

```bibtex
@inproceedings{xie2021simmim,
  title={SimMIM: A Simple Framework for Masked Image Modeling},
  author={Xie, Zhenda and Zhang, Zheng and Cao, Yue and Lin, Yutong and Bao, Jianmin and Yao, Zhuliang and Dai, Qi and Hu, Han},
  booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
