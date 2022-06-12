# BarlowTwins

> [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)

## Abstract

**B**ootstrap **Y**our **O**wn **L**atent (BYOL) is a new approach to self-supervised image representation learning. BYOL relies on two neural networks, referred to as online and target networks, that interact and learn from each other. From an augmented view of an image, we train the online network to predict the target network representation of the same image under a different augmented view. At the same time, we update the target network with a slow-moving average of the online network.

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/149720208-5ffbee78-1437-44c7-9ddb-b8caab60d2c3.png" width="800" />
</div>

## Results and Models

This page is based on documents in [MMSelfSup](https://github.com/open-mmlab/mmselfsup).

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are Top-1 (%). We also provide configs on CIFAR-10, CIFAR-100, and ImageNet-100 datasets according to the setting on ImageNet.

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                                       | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [r50_8xb64_accu8_cos_lr4_8_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/imagenet/r50_8xb64_accu8_cos_lr4_8_fp16_ep200.py) | feature5   | 86.31 | 45.37 | 56.83 | 68.47 | 74.12 | 78.30 | 81.53 | 83.56 | 84.73 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep90.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_mhead_sz224_4xb64_step_ep90.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [r50_linear_sz224_4xb64_step_ep100.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_linear_sz224_4xb64_step_ep100.py) for details of config.

| Self-Supervised Config                                                                                                                                       | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------- | -------- | -------- | -------- | ------- |
| [r50_8xb64_accu8_cos_lr4_8_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/imagenet/r50_8xb64_accu8_cos_lr4_8_fp16_ep200.py) | 15.16    | 35.26    | 47.77    | 63.10    | 71.21    | 71.72   |
| [r50_8xb64_accu8_cos_lr4_8_fp16_ep300](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/imagenet/r50_8xb64_accu8_cos_lr4_8_fp16_ep300.py) | 15.41    | 35.15    | 47.77    | 62.59    | 71.85    | 71.88   |

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep28](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/place205/r50_mhead_sz224_4xb64_step_ep28.py) for details of config.

| Self-Supervised Config                                                                                                                                       | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------- | -------- | -------- | -------- |
| [r50_8xb64_accu8_cos_lr4_8_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/imagenet/r50_8xb64_accu8_cos_lr4_8_fp16_ep200.py) | 21.25    | 36.55    | 43.66    | 50.74    | 53.82    |
| [r50_8xb64_accu8_cos_lr4_8_fp16_ep300](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/imagenet/r50_8xb64_accu8_cos_lr4_8_fp16_ep300.py) | 21.18    | 36.68    | 43.42    | 51.04    | 54.06    |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k_voc0712.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k_voc0712.py) for details of config.

| Self-Supervised Config                                                                                                                                       | AP50  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| [r50_8xb64_accu8_cos_lr4_8_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/imagenet/r50_8xb64_accu8_cos_lr4_8_fp16_ep200.py) | 80.35 |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x_coco.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py) for details of config.

| Self-Supervised Config                                                                                                                                       | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [r50_8xb64_accu8_cos_lr4_8_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/imagenet/r50_8xb64_accu8_cos_lr4_8_fp16_ep200.py) | 40.9 | 61.0      | 44.6      | 36.8      | 58.1       | 39.5       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [fcn_r50-d8_512x512_20k_voc12aug.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py) for details of config.

| Self-Supervised Config                                                                                                                                       | mIOU  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| [r50_8xb64_accu8_cos_lr4_8_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/imagenet/r50_8xb64_accu8_cos_lr4_8_fp16_ep200.py) | 67.16 |

## Citation

```bibtex
@inproceedings{grill2020bootstrap,
  title={Bootstrap your own latent: A new approach to self-supervised learning},
  author={Grill, Jean-Bastien and Strub, Florian and Altch{\'e}, Florent and Tallec, Corentin and Richemond, Pierre H and Buchatskaya, Elena and Doersch, Carl and Pires, Bernardo Avila and Guo, Zhaohan Daniel and Azar, Mohammad Gheshlaghi and others},
  booktitle={NeurIPS},
  year={2020}
}
```
