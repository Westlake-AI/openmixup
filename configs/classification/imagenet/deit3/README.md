# DeiT III

> [DeiT III: Revenge of the ViT](https://arxiv.org/abs/2204.07118)

## Abstract

A Vision Transformer (ViT) is a simple neural architecture amenable to serve several computer vision tasks. It has limited built-in architectural priors, in contrast to more recent architectures that incorporate priors either about the input data or of specific tasks. Recent works show that ViTs benefit from self-supervised pre-training, in particular BerT-like pre-training like BeiT. In this paper, we revisit the supervised training of ViTs. Our procedure builds upon and simplifies a recipe introduced for training ResNet-50. It includes a new simple data-augmentation procedure with only 3 augmentations, closer to the practice in self-supervised learning. Our evaluations on Image classification (ImageNet-1k with and without pre-training on ImageNet-21k), transfer learning and semantic segmentation show that our procedure outperforms by a large margin previous fully supervised training recipes for ViT. It also reveals that the performance of our ViT trained with supervision is comparable to that of more recent architectures. Our results could serve as better baselines for recent self-supervised approaches demonstrated on ViT.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/195295370-b91c23de-5e56-429f-a6f2-0a793777cdc6.png" width="80%"/>
<img src="https://user-images.githubusercontent.com/44519745/195295088-2491d010-85b8-4f09-8718-bbd48ceb56ee.png" width="80%"/>
</div>

## Results and models

### ImageNet-1k

Notice that DeiT3 models are first trained on small resolutions (224x224, 192x192, 160x160) and then fine-tuned 20 epochs on 224x224 resolutions for the final results on ImageNet-1K. Moreover, DeiT3 models are pre-trained on ImageNet-21K and fine-tuned with 384x384 resolutions on ImageNet-1K.

|   Model   | Pre-train resolution | fine-tune resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                   Pre-train config                  |                Fine-tune config                |                                                        Download                                                       |
|:---------:|:--------------------:|:--------------------:|:---------:|:--------:|:---------:|:---------:|:---------------------------------------------------:|:----------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|
| DeiT3-S\* |        224x224       |        224x224       |   22.06   |   4.61   |   81.35   |   95.31   |    [config](./deit3_small_sz224_8xb256_ep800.py)    |  [config](./deit3_small_sz224_8xb256_ep800.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-small-p16_3rdparty_in1k_20221008-0f7c70cf.pth) |
| DeiT3-B\* |        192x192       |        224x224       |   86.59   |   17.58  |   83.80   |   96.55   |  [config](./deit3_base_sz192_8xb128_accu2_ep800.py) | [config](./deit3_base_sz224_ft_4xb128_ep20.py) |  [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-base-p16_3rdparty_in1k_20221008-60b8c8bf.pth) |
| DeiT3-L\* |        192x192       |        224x224       |   304.37  |   61.60  |   84.87   |   97.01   | [config](./deit3_large_sz192_8xb128_accu2_ep800.py) | [config](./deit3_large_sz224_ft_8xb64_ep20.py) | [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-large-p16_3rdparty_in1k_20221009-03b427ea.pth) |
| DeiT3-H\* |        160x160       |        224x224       |   632.13  |  167.40  |   85.21   |   97.36   |  [config](./deit3_huge_sz160_8xb64_accu4_ep800.py)  |  [config](./deit3_huge_sz224_ft_8xb64_ep20.py) |  [model](https://download.openmmlab.com/mmclassification/v0/deit3/deit3-huge-p14_3rdparty_in1k_20221009-e107bcb7.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/deit). The config files of these models are only for validation and we don't ensure these config files' training accuracy.*

## Citation

```
@article{Touvron2022DeiTIR,
  title={DeiT III: Revenge of the ViT},
  author={Hugo Touvron and Matthieu Cord and Herve Jegou},
  journal={arXiv preprint arXiv:2204.07118},
  year={2022},
}
```
