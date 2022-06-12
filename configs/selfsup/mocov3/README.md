# MoCo v3

> [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057)

## Abstract

This paper does not describe a novel method. Instead, it studies a straightforward, incremental, yet must-know baseline given the recent progress in computer vision: self-supervised learning for Vision Transformers (ViT). While the training recipes for standard convolutional networks have been highly mature and robust, the recipes for ViT are yet to be built, especially in the self-supervised scenarios where training becomes more challenging. In this work, we go back to basics and investigate the effects of several fundamental components for training self-supervised ViT. We observe that instability is a major issue that degrades accuracy, and it can be hidden by apparently good results. We reveal that these results are indeed partial failure, and they can be improved when training is made more stable. We benchmark ViT results in MoCo v3 and several other self-supervised frameworks, with ablations in various aspects. We discuss the currently positive evidence as well as challenges and open questions. We hope that this work will provide useful data points and experience for future research.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/151305362-e6e8ea35-b3b8-45f6-8819-634e67083218.png" width="500" />
</div>

## Results and Models

This page is based on documents in [MMSelfSup](https://github.com/open-mmlab/mmselfsup).

### Classification

The classification benchmarks includes 1 downstream task datasets, **ImageNet**. If not specified, the results are Top-1 (%). We also provide configs on CIFAR-10, CIFAR-100, and ImageNet-100 datasets according to the setting on ImageNet.

#### ImageNet Linear Evaluation

The **Linear Evaluation** result is obtained by training a linear head upon the pre-trained backbone. Please refer to [vit_small_p16_linear_sz224_8xb128_cos_ep90.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_small_p16_linear_sz224_8xb128_cos_ep90.py) for details of config.

| Self-Supervised Config                                                                                                                                                                              | Linear Evaluation |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------: |
| [vit_small_8xb64_accu8_cos_fp16_ep300](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov3/imagenet/vit_small_8xb64_accu8_cos_fp16_ep300.py) | 73.19             |

## Citation

```bibtex
@InProceedings{Chen_2021_ICCV,
    title     = {An Empirical Study of Training Self-Supervised Vision Transformers},
    author    = {Chen, Xinlei and Xie, Saining and He, Kaiming},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021}
}
```
