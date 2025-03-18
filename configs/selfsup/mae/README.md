# MAE

> [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

## Abstract

This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3× or more) and improve accuracy. Our scalable approach allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pretraining and shows promising scaling behavior.

<div align="center">
<img src="https://user-images.githubusercontent.com/30762564/150733959-2959852a-c7bd-4d3f-911f-3e8d8839fe67.png" width="40%"/>
</div>

## Models and Benchmarks

<<<<<<< HEAD
Here, we report the results of the model, which is pre-trained on ImageNet-1k
for 400 epochs, the details are below:
=======
Here, we report the results of the model, which is pre-trained on ImageNet-1k for 400 epochs, the details are below:
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)

| Backbone | Pre-train epoch | Fine-tuning Top-1 | Pre-train Config | Fine-tuning Config | Download |
| :----: | :----: | :----: | :----: | :----: | :----: |
| ViT-B/16 | 400 | 83.3 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mae/imagenet/vit_base_dec8_dim512_8xb512_cos_fp16_ep400.py) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_base_p16_swin_ft_mae_sz224_8xb128_cos_ep100.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k_20220825-bc79e40b.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k_20220628_200815.json) |
| ViT-B/16 | 800 | 83.3 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mae/imagenet/vit_base_dec8_dim512_8xb512_cos_fp16_ep800.py) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_base_p16_swin_ft_mae_sz224_8xb128_cos_ep100.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220825-5d81fbc4.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220718_134405.json) |
| ViT-B/16 | 1600 | 83.5 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mae/imagenet/vit_base_dec8_dim512_8xb512_cos_fp16_ep1600.py) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_base_p16_swin_ft_mae_sz224_8xb128_cos_ep100.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220815_103458.json) |
| ViT-L/16 | 400 | 85.2 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mae/imagenet/vit_large_dec8_dim512_8xb256_accu2_cos_fp16_ep400.py) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_large_p16_swin_ft_mae_sz224_8xb128_cos_ep100.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k_20220825-b11d0425.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k_20220726_202204.json) |
| ViT-L/16 | 800 | 85.4 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mae/imagenet/vit_large_dec8_dim512_8xb256_accu2_cos_fp16_ep800.py) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_large_p16_swin_ft_mae_sz224_8xb128_cos_ep100.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k_20220825-df72726a.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k_20220804_104018.json) |
| ViT-L/16 | 1600 | 85.7 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mae/imagenet/vit_large_dec8_dim512_8xb256_accu2_cos_fp16_ep1600.py) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_large_p16_swin_ft_mae_sz224_8xb128_cos_ep100.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220806_210725.json) |
| ViT-H/14 | 1600 | 86.9 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mae/imagenet/vit_huge_dec8_dim512_8xb128_accu4_cos_fp16_ep1600.py) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_huge_p14_swin_ft_mae_sz224_8xb128_cos_ep50.py) | [model](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220916-ff848775.pth) \| [log](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220814_135241.json) |

## Citation

```bibtex
@inproceedings{he2022MAE,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll'ar and Ross B. Girshick},
  booktitle={CVPR},
  year={2022}
}
```
