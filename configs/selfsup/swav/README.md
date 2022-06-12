# SwAV

> [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882)

## Abstract

Unsupervised image representations have significantly reduced the gap with supervised pretraining, notably with the recent achievements of contrastive learning methods. These contrastive methods typically work online and rely on a large number of explicit pairwise feature comparisons, which is computationally challenging. In this paper, we propose an online algorithm, SwAV, that takes advantage of contrastive methods without requiring to compute pairwise comparisons. Specifically, our method simultaneously clusters the data while enforcing consistency between cluster assignments produced for different augmentations (or “views”) of the same image, instead of comparing features directly as in contrastive learning. Simply put, we use a “swapped” prediction mechanism where we predict the code of a view from the representation of another view. Our method can be trained with large and small batches and can scale to unlimited amounts of data. Compared to previous contrastive methods, our method is more memory efficient since it does not require a large memory bank or a special momentum network. In addition, we also propose a new data augmentation strategy, multi-crop, that uses a mix of views with different resolutions in place of two full-resolution views, without increasing the memory or compute requirements.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149724517-9f1e7bdf-04c7-43e3-92f4-2b8fc1399123.png" width="500" />
</div>

## Results and Models

This page is based on documents in [MMSelfSup](https://github.com/open-mmlab/mmselfsup).

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                                                              | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/swav/imagenet/r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200.py) | feature5   | 87.00 | 44.68 | 55.41 | 67.64 | 73.67 | 78.14 | 81.58 | 83.98 | 85.15 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep90.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_mhead_sz224_4xb64_step_ep90.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [r50_linear_sz224_4xb64_step_ep100.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_linear_sz224_4xb64_step_ep100.py) for details of config.

| Self-Supervised Config                                                                                                                                                              | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/swav/imagenet/r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200.py) | 16.98    | 34.96    | 49.26    | 65.98    | 70.74    | 70.47   |

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep28](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/place205/r50_mhead_sz224_4xb64_step_ep28.py) for details of config.

| Self-Supervised Config                                                                                                                                                              | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/swav/imagenet/r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200.py) | 23.33    | 35.45    | 43.13    | 51.98    | 55.09    |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k_voc0712.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k_voc0712.py) for details of config.

| Self-Supervised Config                                                                                                                                                              | AP50  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/swav/imagenet/r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200.py) | 77.64 |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x_coco.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py) for details of config.

| Self-Supervised Config                                                                                                                                                              | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/swav/imagenet/r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200.py) | 40.2     | 60.5      | 43.9      | 36.3      | 57.5       | 38.8       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [fcn_r50-d8_512x512_20k_voc12aug.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py) for details of config.

| Self-Supervised Config                                                                                                                                                              | mIOU  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/swav/imagenet/r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200.py) | 63.73 |

## Citation

```bibtex
@article{caron2020unsupervised,
  title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  booktitle={NeurIPS},
  year={2020}
}
```
