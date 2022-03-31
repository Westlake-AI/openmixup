# Model Zoo

**Current results of self-supervised learning benchmarks are based on [MMSelfSup](https://github.com/open-mmlab/mmselfsup) and [solo-learn](https://github.com/vturrisi/solo-learn). We will rerun the experiments and update more reliable results soon!**

<details open>
<summary>Supported sample mixing policies</summary>

- [x] [Relative-Loc [ICCV 2015]](https://arxiv.org/abs/1505.05192)
- [x] [Rotation-Pred [ICLR 2018]](https://arxiv.org/abs/1803.07728)
- [x] [DeepCluster [ECCV 2018]](https://arxiv.org/abs/1807.05520)
- [x] [NPID [CVPR 2018]](https://arxiv.org/abs/1805.01978)
- [x] [ODC [CVPR 2020]](https://arxiv.org/abs/2006.10645)
- [x] [MoCo [CVPR 2020]](https://arxiv.org/abs/1911.05722)
- [x] [MoCo.V2 [Arxiv 2020]](https://arxiv.org/abs/2003.04297)
- [x] [MoCo.V3 [Arxiv 2021]](https://arxiv.org/abs/2104.02057)
- [x] [SimCLR [ICML 2020]](https://arxiv.org/abs/2002.05709)
- [x] [BYOL [NIPS 2020]](https://arxiv.org/abs/2006.07733)
- [x] [SwAV [NIPS 2020]](https://arxiv.org/abs/2006.09882)
- [x] [DenseCL [CVPR 2021]](https://arxiv.org/abs/2011.09157)
- [x] [SimSiam [CVPR 2021]](https://arxiv.org/abs/2011.10566)
- [x] [MAE [CVPR 2022]](https://arxiv.org/abs/2111.06377)

</details>

## ImageNet-1k pre-trained models
**Note**
* If not specifically indicated, the testing GPUs are NVIDIA Tesla V100.
* The table records the implementors who implemented the methods (either by themselves or refactoring from other repos), and the experimenters who performed experiments and reproduced the results. The experimenters should be responsible for the evaluation results on all the benchmarks, and the implementors should be responsible for the implementation as well as the results; If the experimenter is not indicated, an implementator is the experimenter by default.

| Methods       | Remarks     | Batch size | Epochs | Linear |
|---------------|-------------|------------|--------|--------|
| ImageNet      | torchvision | -          | -      | 76.17  |
| Random        | kaiming     | -          | -      | 4.35   |
| Relative-Loc  | ResNet-50   | 512        | 70     | 38.83  |
| Rotation-Pred | ResNet-50   | 128        | 70     | 47.01  |
| DeepCluster   | ResNet-50   | 512        | 200    | 46.92  |
| NPID          | ResNet-50   | 256        | 200    | 56.60  |
| ODC           | ResNet-50   | 512        | 440    | 53.42  |
| MoCo          | ResNet-50   | 256        | 200    | 61.02  |
| MoCo.V2       | ResNet-50   | 256        | 200    | 67.69  |
| MoCo.V3       | ViT-small   | 4096       | 400    |        |
| SimCLR        | ResNet-50   | 4096       | 200    |        |
| BYOL          | ResNet-50   | 4096       | 200    | 67.10  |
| SwAV          | ResNet-50   | 4096       | 200    |        |
| DenseCL       | ResNet-50   | 256        | 200    |        |
| SimSiam       | ResNet-50   | 512        | 200    |        |
| MAE           | ViT-base    | 4096       | 800    |        |


## Benchmarks

### VOC07 SVM & SVM Low-shot


### ImageNet Linear Classification

**Note**
* Config: `configs/benchmarks/linear_classification/imagenet/r50_multihead.py` for ImageNet (Multi) and `configs/benchmarks/linear_classification/imagenet/r50_last.py` for ImageNet (Last).
* For DeepCluster, use the corresponding one with `_sobel`.
* ImageNet (Multi) evaluates features in around 9k dimensions from different layers. Top-1 result of the last epoch is reported.
* ImageNet (Last) evaluates the last feature after global average pooling, e.g., 2048 dimensions for resnet50. The best top-1 result among all epochs is reported.
* Usually, we report the best result from ImageNet (Multi) and ImageNet (Last) to ensure fairness, since different methods achieve their best performance on different layers.


### Places205 Linear Classification

**Note**
* Config: `configs/benchmarks/linear_classification/places205/r50_multihead.py`.
* For DeepCluster, use the corresponding one with `_sobel`.
* Places205 evaluates features in around 9k dimensions from different layers. Top-1 result of the last epoch is reported.


### ImageNet Semi-Supervised Classification

**Note**
* In this benchmark, the necks or heads are removed and only the backbone CNN is evaluated by appending a linear classification head. All parameters are fine-tuned.
* Config: under `configs/benchmarks/semi_classification/imagenet_1percent/` for 1% data, and `configs/benchmarks/semi_classification/imagenet_10percent/` for 10% data.
* When training with 1% ImageNet, we find hyper-parameters especially the learning rate greatly influence the performance. Hence, we prepare a list of settings with the base learning rate from \{0.001, 0.01, 0.1\} and the learning rate multiplier for the head from \{1, 10, 100\}. We choose the best performing setting for each method.
* Please use `--deterministic` in this benchmark.


### PASCAL VOC07+12 Object Detection

**Note**
* This benchmark follows the evluation protocols set up by MoCo.
* Config: `benchmarks/detection/configs/pascal_voc_R_50_C4_24k_moco.yaml`.
* Please follow [here](GETTING_STARTED.md#voc0712--coco17-object-detection) to run the evaluation.


### COCO2017 Object Detection

**Note**
* This benchmark follows the evluation protocols set up by MoCo.
* Config: `benchmarks/detection/configs/coco_R_50_C4_2x_moco.yaml`.
* Please follow [here](GETTING_STARTED.md#voc0712--coco17-object-detection) to run the evaluation.
