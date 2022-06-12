# NPID

> [Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://arxiv.org/abs/1805.01978)

## Abstract

Neural net classifiers trained on data with annotated class labels can also capture apparent visual similarity among categories without being directed to do so. We study whether this observation can be extended beyond the conventional domain of supervised learning: Can we learn a good feature representation that captures apparent similar- ity among instances, instead of classes, by merely asking the feature to be discriminative of individual instances? We formulate this intuition as a non-parametric classification problem at the instance-level, and use noise-contrastive estimation to tackle the computational challenges imposed by the large number of instance classes. Our experimental results demonstrate that, under unsupervised learning settings, our method surpasses the state-of-the-art on ImageNet classification by a large margin. Our method is also remarkable for consistently improving test performance with more training data and better network architectures. By fine-tuning the learned feature, we further obtain competitive results for semi-supervised learning and object detection tasks. Our non-parametric model is highly compact: With 128 features per image, our method requires only 600MB storage for a million images, enabling fast nearest neighbour retrieval at the run time.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149722257-1651c283-ac68-4cdc-90e6-970d820529af.png" width="800" />
</div>

## Results and Models

This page is based on documents in [MMSelfSup](https://github.com/open-mmlab/mmselfsup).

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are  Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                         | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [r50_4xb64_step_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/npid/imagenet/r50_4xb64_step_ep200.py) | feature5   | 76.75 | 26.96 | 35.37 | 44.48 | 53.89 | 60.39 | 66.41 | 71.48 | 73.39 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep90.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_mhead_sz224_4xb64_step_ep90.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [r50_linear_sz224_4xb64_step_ep100.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_linear_sz224_4xb64_step_ep100.py) for details of config.

| Self-Supervised Config                                                                                                                         | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [r50_4xb64_step_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/npid/imagenet/r50_4xb64_step_ep200.py) | 14.68    | 31.98    | 42.85    | 56.95    | 58.41    | 57.97   |

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep28](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/place205/r50_mhead_sz224_4xb64_step_ep28.py) for details of config.

| Self-Supervised Config                                                                                                                         | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [r50_4xb64_step_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/npid/imagenet/r50_4xb64_step_ep200.py) | 19.98    | 34.86    | 41.59    | 48.43    | 48.71    |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k_voc0712.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k_voc0712.py) for details of config.

| Self-Supervised Config                                                                                                                         | AP50  |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [r50_4xb64_step_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/npid/imagenet/r50_4xb64_step_ep200.py) | 79.52 |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x_coco.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py) for details of config.

| Self-Supervised Config                                                                                                                         | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [r50_4xb64_step_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/npid/imagenet/r50_4xb64_step_ep200.py) | 38.5     | 57.7      | 42.0      | 34.6      | 54.8       | 37.1       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [fcn_r50-d8_512x512_20k_voc12aug.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py) for details of config.

| Self-Supervised Config                                                                                                                         | mIOU  |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [r50_4xb64_step_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/npid/imagenet/r50_4xb64_step_ep200.py) | 65.45 |

## Citation

```bibtex
@inproceedings{wu2018unsupervised,
  title={Unsupervised feature learning via non-parametric instance discrimination},
  author={Wu, Zhirong and Xiong, Yuanjun and Yu, Stella X and Lin, Dahua},
  booktitle={CVPR},
  year={2018}
}
```
