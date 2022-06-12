# MoCo v2

> [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297)

## Abstract

Contrastive unsupervised learning has recently shown encouraging progress, e.g., in Momentum Contrast (MoCo) and SimCLR. In this note, we verify the effectiveness of two of SimCLR’s design improvements by implementing them in the MoCo framework. With simple modifications to MoCo—namely, using an MLP projection head and more data augmentation—we establish stronger baselines that outperform SimCLR and do not require large training batches. We hope this will make state-of-the-art unsupervised learning research more accessible.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149720067-b65e5736-d425-45b3-93ed-6f2427fc6217.png" width="500" />
</div>

## Results and Models

This page is based on documents in [MMSelfSup](https://github.com/open-mmlab/mmselfsup).

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are  Top-1 (%). We also provide configs on CIFAR-10, CIFAR-100, and ImageNet-100 datasets according to the setting on ImageNet.

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                           | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [r50_4xb64_cos_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov2/imagenet/r50_4xb64_cos_fp16_ep200.py) | feature5   | 84.04 | 43.14 | 53.29 | 65.34 | 71.03 | 75.42 | 78.48 | 80.88 | 82.23 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep90.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_mhead_sz224_4xb64_step_ep90.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [r50_linear_sz224_4xb64_step_ep100.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_linear_sz224_4xb64_step_ep100.py) for details of config.

| Self-Supervised Config                                                                                                                           | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------- | -------- | -------- | -------- | ------- |
| [r50_4xb64_cos_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov2/imagenet/r50_4xb64_cos_fp16_ep200.py) | 15.96    | 34.22    | 45.78    | 61.11    | 66.24    | 67.58   |

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep28](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/place205/r50_mhead_sz224_4xb64_step_ep28.py) for details of config.

| Self-Supervised Config                                                                                                                           | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------- | -------- | -------- | -------- |
| [r50_4xb64_cos_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov2/imagenet/r50_4xb64_cos_fp16_ep200.py) | 20.92    | 35.72    | 42.62    | 49.79    | 52.25    |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k_voc0712.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k_voc0712.py) for details of config.

| Self-Supervised Config                                                                                                                           | AP50  |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| [r50_4xb64_cos_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov2/imagenet/r50_4xb64_cos_fp16_ep200.py) | 81.06 |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x_coco.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py) for details of config.

| Self-Supervised Config                                                                                                                           | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [r50_4xb64_cos_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov2/imagenet/r50_4xb64_cos_fp16_ep200.py) | 40.2     | 59.7      | 44.2      | 36.1      | 56.7       | 38.8       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [fcn_r50-d8_512x512_20k_voc12aug.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py) for details of config.

| Self-Supervised Config                                                                                                                           | mIOU  |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| [r50_4xb64_cos_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov2/imagenet/r50_4xb64_cos_fp16_ep200.py) | 67.55 |

## Citation

```bibtex
@article{chen2020improved,
  title={Improved baselines with momentum contrastive learning},
  author={Chen, Xinlei and Fan, Haoqi and Girshick, Ross and He, Kaiming},
  journal={arXiv preprint arXiv:2003.04297},
  year={2020}
}
```
