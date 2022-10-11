# MaskFeat

> [Masked Feature Prediction for Self-Supervised Visual Pre-Training](https://arxiv.org/abs/2112.09133)

## Abstract

We present Masked Feature Prediction (MaskFeat) for self-supervised pre-training of video models. Our approach first randomly masks out a portion of the input sequence and then predicts the feature of the masked regions. We study five different types of features and find Histograms of Oriented Gradients (HOG), a hand-crafted feature descriptor, works particularly well in terms of both performance and efficiency. We observe that the local contrast normalization in HOG is essential for good results, which is in line with earlier work using HOG for visual recognition. Our approach can learn abundant visual knowledge and drive large-scale Transformer-based models. Without using extra model weights or supervision, MaskFeat pre-trained on unlabeled videos achieves unprecedented results of 86.7% with MViT-L on Kinetics-400, 88.3% on Kinetics-600, 80.4% on Kinetics-700, 38.8 mAP on AVA, and 75.0% on SSv2. MaskFeat further generalizes to image input, which can be interpreted as a video with a single frame and obtains competitive results on ImageNet.

## Models and Benchmarks

Here, we report the results provided in the [original repo](https://github.com/facebookresearch/SlowFast), which is pre-trained 300/800-epoch with ViT-Base on ImageNet-1k.

| Backbone | Pre-train epoch | Fine-tuning Top-1 |                                                                   Pre-train Config                                                                   |                                                                          Fine-tuning Config                                                                         |   Download   |
|:--------:|:---------------:|:-----------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------:|
| ViT-Base |       300       |        83.1       | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/maskfeat/imagenet/vit_base_hog_108_sz224_8xb128_accu2_cos_fp16_ep300.py) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_base_p16_swin_ft_simmim_sz224_4xb128_accu2_cos_ep100.py) | model \| log |
| ViT-Base |       300       |        84.0       | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/maskfeat/imagenet/vit_base_hog_108_sz224_8xb128_accu2_cos_fp16_ep800.py) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_base_p16_swin_ft_simmim_sz224_4xb128_accu2_cos_ep100.py) | model \| log |

## Citation

```bibtex
@article{Wei2021MaskFeat,
  title={Masked Feature Prediction for Self-Supervised Visual Pre-Training},
  author={Wei, Chen and Fan, Haoqi and Xie, Saining and Wu, Chao-Yuan and Yuille, Alan and Feichtenhofer, Christoph},
  journal={ArXiv},
  url={https://arxiv.org/abs/2112.09133},
  year={2021}
}
```
