# SimSiam

> [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)

## Abstract

Siamese networks have become a common structure in various recent models for unsupervised visual representation learning. These models maximize the similarity between two augmentations of one image, subject to certain conditions for avoiding collapsing solutions. In this paper, we report surprising empirical results that simple Siamese networks can learn meaningful representations even using none of the following: (i) negative sample pairs, (ii) large batches, (iii) momentum encoders. Our experiments show that collapsing solutions do exist for the loss and structure, but a stop-gradient operation plays an essential role in preventing collapsing. We provide a hypothesis on the implication of stop-gradient, and further show proof-of-concept experiments verifying it. Our “SimSiam” method achieves competitive results on ImageNet and downstream tasks. We hope this simple baseline will motivate people to rethink the roles of Siamese architectures for unsupervised representation learning.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149724180-bc7bac6a-fcb8-421e-b8f1-9550c624d154.png" width="500" />
</div>

## Results and Models

This page is based on documents in [MMSelfSup](https://github.com/open-mmlab/mmselfsup).

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                             | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [r50_4xb64_cos_lr0_05_ep100](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep100.py) | feature5   | 84.64 | 39.65 | 49.86 | 62.48 | 69.50 | 74.48 | 78.31 | 81.06 | 82.56 |
| [r50_4xb64_cos_lr0_05_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep200.py) | feature5   | 85.20 | 39.85 | 50.44 | 63.73 | 70.93 | 75.74 | 79.42 | 82.02 | 83.44 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep90.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_mhead_sz224_4xb64_step_ep90.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [r50_linear_sz224_4xb64_step_ep100.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_linear_sz224_4xb64_step_ep100.py) for details of config.

| Self-Supervised Config                                                                                                                             | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [r50_4xb64_cos_lr0_05_ep100](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep100.py) | 16.27    | 33.77    | 45.80    | 60.83    | 68.21    | 68.28   |
| [r50_4xb64_cos_lr0_05_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep200.py) | 15.57    | 37.21    | 47.28    | 62.21    | 69.85    | 69.84   |

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep28](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/place205/r50_mhead_sz224_4xb64_step_ep28.py) for details of config.

| Self-Supervised Config                                                                                                                             | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [r50_4xb64_cos_lr0_05_ep100](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep100.py) | 21.32    | 35.66    | 43.05    | 50.79    | 53.27    |
| [r50_4xb64_cos_lr0_05_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep200.py) | 21.17    | 35.85    | 43.49    | 50.99    | 54.10    |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k_voc0712.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k_voc0712.py) for details of config.

| Self-Supervised Config                                                                                                                             | AP50  |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [r50_4xb64_cos_lr0_05_ep100](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep100.py) | 79.80 |
| [r50_4xb64_cos_lr0_05_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep200.py) | 79.85 |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x_coco.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py) for details of config.

| Self-Supervised Config                                                                                                                             | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [r50_4xb64_cos_lr0_05_ep100](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep100.py) | 38.6     | 57.6      | 42.3      | 34.6      | 54.8       | 36.9       |
| [r50_4xb64_cos_lr0_05_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep200.py) | 38.8     | 58.0      | 42.3      | 34.9      | 55.3       | 37.6       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [fcn_r50-d8_512x512_20k_voc12aug.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py) for details of config.

| Self-Supervised Config                                                                                                                             | mIOU  |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [r50_4xb64_cos_lr0_05_ep100](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep100.py) | 48.35 |
| [r50_4xb64_cos_lr0_05_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep200.py) | 46.27 |

## Citation

```bibtex
@inproceedings{chen2021exploring,
  title={Exploring simple siamese representation learning},
  author={Chen, Xinlei and He, Kaiming},
  booktitle={CVPR},
  year={2021}
}
```
