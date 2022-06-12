# SimCLR

> [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

## Abstract

This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectures or a memory bank. In order to understand what enables the contrastive prediction tasks to learn useful representations, we systematically study the major components of our framework. We show that (1) composition of data augmentations plays a critical role in defining effective predictive tasks, (2) introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning. By combining these findings, we are able to considerably outperform previous methods for self-supervised and semi-supervised learning on ImageNet. A linear classifier trained on self-supervised representations learned by SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art, matching the performance of a supervised ResNet-50.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149723851-cf5f309e-d891-454d-90c0-e5337e5a11ed.png" width="400" />
</div>

## Results and Models

This page is based on documents in [MMSelfSup](https://github.com/open-mmlab/mmselfsup).

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are Top-1 (%). We also provide configs on CIFAR-10, CIFAR-100, and ImageNet-100 datasets according to the setting on ImageNet.

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                           | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96 |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ---- |
| [r50_8xb64_cos_lr0_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simclr/imagenet/r50_8xb64_cos_lr0_6_fp16_ep200.py) | feature5   | 79.98 | 35.02 | 42.79 | 54.87 | 61.91 | 67.38 | 71.88 | 75.56 | 77.4 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep90.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_mhead_sz224_4xb64_step_ep90.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [r50_linear_sz224_4xb64_step_ep100.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_linear_sz224_4xb64_step_ep100.py) for details of config.

| Self-Supervised Config                                                                                                                               | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [r50_8xb64_cos_lr0_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simclr/imagenet/r50_8xb64_cos_lr0_6_fp16_ep200.py)     | 16.29    | 31.11    | 39.99    | 55.06    | 62.91    | 62.56   |
| [r50_16xb256_cos_lr4_8_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simclr/imagenet/r50_16xb256_cos_lr4_8_fp16_ep200.py) | 15.44    | 31.47    | 41.83    | 59.44    | 66.41    | 66.66   |

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep28](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/place205/r50_mhead_sz224_4xb64_step_ep28.py) for details of config.

| Self-Supervised Config                                                                                                                           | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------- | -------- | -------- | -------- |
| [r50_8xb64_cos_lr0_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simclr/imagenet/r50_8xb64_cos_lr0_6_fp16_ep200.py) | 20.60    | 33.62    | 38.86    | 45.25    | 50.91    |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k_voc0712.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k_voc0712.py) for details of config.

| Self-Supervised Config                                                                                                                           | AP50  |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| [r50_8xb64_cos_lr0_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simclr/imagenet/r50_8xb64_cos_lr0_6_fp16_ep200.py) | 79.38 |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x_coco.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py) for details of config.

| Self-Supervised Config                                                                                                                           | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [r50_8xb64_cos_lr0_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simclr/imagenet/r50_8xb64_cos_lr0_6_fp16_ep200.py) | 38.7     | 58.1      | 42.4      | 34.9      | 55.3       | 37.5       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [fcn_r50-d8_512x512_20k_voc12aug.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py) for details of config.

| Self-Supervised Config                                                                                                                           | mIOU  |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| [r50_8xb64_cos_lr0_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simclr/imagenet/r50_8xb64_cos_lr0_6_fp16_ep200.py) | 64.03 |

## Citation

```bibtex
@inproceedings{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={ICML},
  year={2020},
}
```
