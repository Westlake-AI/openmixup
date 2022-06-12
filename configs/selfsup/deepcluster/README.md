# DeepCluster

> [Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520)

## Abstract

Clustering is a class of unsupervised learning methods that has been extensively applied and studied in computer vision. Little work has been done to adapt it to the end-to-end training of visual features on large scale datasets. In this work, we present DeepCluster, a clustering method that jointly learns the parameters of a neural network and the cluster assignments of the resulting features. DeepCluster iteratively groups the features with a standard clustering algorithm, k-means, and uses the subsequent assignments as supervision to update the weights of the network.

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/149720586-5bfd213e-0638-47fc-b48a-a16689190e17.png" width="700" />
</div>

## Results and Models

This page is based on documents in [MMSelfSup](https://github.com/open-mmlab/mmselfsup).

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are Top-1 (%). We also provide configs on CIFAR-10, CIFAR-100, and ImageNet-100 datasets according to the setting on ImageNet.

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                                                   | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [r50_sobel_8xb64_step_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/deepcluster/imagenet/r50_sobel_8xb64_step_ep200.py) | feature5   | 74.26 | 29.37 | 37.99 | 45.85 | 55.57 | 62.48 | 66.15 | 70.00 | 71.37 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep90.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_mhead_sz224_4xb64_step_ep90.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [r50_linear_sz224_4xb64_step_ep100.py](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_linear_sz224_4xb64_step_ep100.py) for details of config.

| Self-Supervised Config                                                                                                                                                   | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------- | -------- | -------- | -------- | ------- |
| [r50_sobel_8xb64_step_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/deepcluster/imagenet/r50_sobel_8xb64_step_ep200.py) | 12.78    | 30.81    | 43.88    | 57.71    | 51.68    | 46.92   |

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [r50_mhead_sz224_4xb64_step_ep28](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/place205/r50_mhead_sz224_4xb64_step_ep28.py) for details of config.

| Self-Supervised Config                                                                                                                                                   | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------- | -------- | -------- | -------- |
| [r50_sobel_8xb64_step_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/deepcluster/imagenet/r50_sobel_8xb64_step_ep200.py) | 18.80    | 33.93    | 41.44    | 47.22    | 42.61    |

## Citation

```bibtex
@inproceedings{caron2018deep,
  title={Deep clustering for unsupervised learning of visual features},
  author={Caron, Mathilde and Bojanowski, Piotr and Joulin, Armand and Douze, Matthijs},
  booktitle={ECCV},
  year={2018}
}
```
