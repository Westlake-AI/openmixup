# DenseCL

> [Dense Contrastive Learning for Self-Supervised Visual Pre-Training](https://arxiv.org/abs/2011.09157)

## Abstract

To date, most existing self-supervised learning methods are designed and optimized for image classification. These pre-trained models can be sub-optimal for dense prediction tasks due to the discrepancy between image-level prediction and pixel-level prediction. To fill this gap, we aim to design an effective, dense self-supervised learning method that directly works at the level of pixels (or local features) by taking into account the correspondence between local features. We present dense contrastive learning (DenseCL), which implements self-supervised learning by optimizing a pairwise contrastive (dis)similarity loss at the pixel level between two views of input images.

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/149721111-bab03a6d-a30d-418e-b338-43c3689cfc65.png" width="900" />
</div>

## Results and Models

This page is based on documents in [MMSelfSup](https://github.com/open-mmlab/mmselfsup).

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are  Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                             | Best Layer | SVM  | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [r50_4xb64_cos_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/densecl/imagenet/r50_4xb64_cos_fp16_ep200.py) | feature5   | 82.5 | 42.68 | 50.64 | 61.74 | 68.17 | 72.99 | 76.07 | 79.19 | 80.55 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep90.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_mhead_sz224_4xb64_step_ep90.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [r50_linear_sz224_4xb64_step_ep100.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_linear_sz224_4xb64_step_ep100.py) for details of config.

| Self-Supervised Config                                                                                                                             | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [r50_4xb64_cos_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/densecl/imagenet/r50_4xb64_cos_fp16_ep200.py) | 15.86    | 35.47    | 49.46    | 64.06    | 62.95    | 63.34   |

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep28](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/place205/r50_mhead_sz224_4xb64_step_ep28.py) for details of config.

| Self-Supervised Config                                                                                                                             | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [r50_4xb64_cos_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/densecl/imagenet/r50_4xb64_cos_fp16_ep200.py) | 21.32    | 36.20    | 43.97    | 51.04    | 50.45    |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k_voc0712.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k_voc0712.py) for details of config.

| Self-Supervised Config                                                                                                                             | AP50  |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [r50_4xb64_cos_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/densecl/imagenet/r50_4xb64_cos_fp16_ep200.py) | 82.14 |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x_coco.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py) for details of config.

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [fcn_r50-d8_512x512_20k_voc12aug.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py) for details of config.

| Self-Supervised Config                                                                                                                             | mIOU  |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [r50_4xb64_cos_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/densecl/imagenet/r50_4xb64_cos_fp16_ep200.py) | 69.47 |

## Citation

```bibtex
@inproceedings{wang2021dense,
  title={Dense contrastive learning for self-supervised visual pre-training},
  author={Wang, Xinlong and Zhang, Rufeng and Shen, Chunhua and Kong, Tao and Li, Lei},
  booktitle={CVPR},
  year={2021}
}
```
