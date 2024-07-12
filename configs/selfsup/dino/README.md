# DINO

> [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)

## Abstract

In this paper, we question if self-supervised learning provides new properties to Vision Transformer (ViT) that stand out compared to convolutional networks (convnets). Beyond the fact that adapting self-supervised methods to this architecture works particularly well, we make the following observations: first, self-supervised ViT features contain explicit information about the semantic segmentation of an image, which does not emerge as clearly with supervised ViTs, nor with convnets. Second, these features are also excellent k-NN classifiers, reaching 78.3% top-1 on ImageNet with a small ViT. Our study also underlines the importance of momentum encoder, multi-crop training, and the use of small patches with ViTs. We implement our findings into a simple self-supervised method, called DINO, which we interpret as a form of self-distillation with no labels. We show the synergy between DINO and ViTs by achieving 80.1% top-1 on ImageNet in linear evaluation with ViT-Base. 

<div align="center">
<img src="https://github.com/user-attachments/assets/ec849125-3816-4411-9142-63edfcdf68fa" width="45%"/>
</div>

## Models and Benchmarks

Here, we report the results of the model, which is pre-trained on ImageNet-1k for 100 epochs based on [official implementation](https://github.com/facebookresearch/dino):
```shell
bash tools/dist_train.sh configs/selfsup/dino/imagenet/vit_base_8xb64_accu8_cos_fp16_ep100.py 8
```

## Citation

```bibtex
@inproceedings{iccv2021dino,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Mathilde Caron and Hugo Touvron and Ishan Misra and Herv'e J'egou and Julien Mairal and Piotr Bojanowski and Armand Joulin},
  journal={2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021},
  pages={9630-9640},
}
```
