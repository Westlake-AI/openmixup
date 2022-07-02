# UniFormer: Unifying Convolution and Self-attention for Visual Recognition

> [UniFormer: Unifying Convolution and Self-attention for Visual Recognition](https://arxiv.org/abs/2201.09450)

## Abstract

It is a challenging task to learn discriminative representation from images and videos, due to large local redundancy and complex global dependency in these visual data. Convolution neural networks (CNNs) and vision transformers (ViTs) have been two dominant frameworks in the past few years. Though CNNs can efficiently decrease local redundancy by convolution within a small neighborhood, the limited receptive field makes it hard to capture global dependency. Alternatively, ViTs can effectively capture long-range dependency via self-attention, while blind similarity comparisons among all the tokens lead to high redundancy. To resolve these problems, we propose a novel Unified transFormer (UniFormer), which can seamlessly integrate the merits of convolution and self-attention in a concise transformer format. Different from the typical transformer blocks, the relation aggregators in our UniFormer block are equipped with local and global token affinity respectively in shallow and deep layers, allowing to tackle both redundancy and dependency for efficient and effective representation learning. Finally, we flexibly stack our UniFormer blocks into a new powerful backbone, and adopt it for various vision tasks from image to video domain, from classification to dense prediction. Without any extra training data, our UniFormer achieves 86.3 top-1 accuracy on ImageNet-1K classification. With only ImageNet-1K pre-training, it can simply achieve state-of-the-art performance in a broad range of downstream tasks, e.g., it obtains 82.9/84.8 top-1 accuracy on Kinetics-400/600, 60.9/71.2 top-1 accuracy on Something-Something V1/V2 video classification tasks, 53.8 box AP and 46.4 mask AP on COCO object detection task, 50.8 mIoU on ADE20K semantic segmentation task, and 77.4 AP on COCO pose estimation task. Code is available at [this https URL](https://github.com/Sense-X/UniFormer).

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/174448564-fc87501f-8d75-4b6c-9545-f1c6cd4043a7.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

<!-- |    Model    |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                               Config                                |                               Download                                |
| :---------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :-----------------------------------------------------------------: | :-------------------------------------------------------------------: |
| UniFormer-S | From scratch |  224x224   |   4.11    |   0.88   |   75.35   |   92.79   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/van/van_tiny_8xb128_fp16_ep300.py) | model / log |
| UniFormer-S | From scratch |  224x224   |   22.86   |   2.52   |   82.90   |   95.63   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/van/van_small_8xb128_fp16_ep310.py) | [model](https://download.openmmlab.com/mmclassification/v0/van/van-small_8xb128_in1k_20220501-17bc91aa.pth) | -->

We follow the original training setting provided by the [official repo](https://github.com/Sense-X/UniFormer). *Note that models with * are converted from [the official repo](https://github.com/Sense-X/UniFormer/blob/main/image_classification).* Based on VAN, we also design and reproduce UniFormer-Tiny.


| Model           | Pretrain    | Resolution | Top-1 | #Param. | FLOPs |
| --------------- | ----------- | ---------- | ----- | ------- | ----- |
| UniFormer-S     | ImageNet-1K | 224x224    | 82.9  | 22M     | 3.6G  |
| UniFormer-S†    | ImageNet-1K | 224x224    | 83.4  | 24M     | 4.2G  |
| UniFormer-B     | ImageNet-1K | 224x224    | 83.9  | 50M     | 8.3G  |
| UniFormer-S+TL  | ImageNet-1K | 224x224    | 83.4  | 22M     | 3.6G  |
| UniFormer-S†+TL | ImageNet-1K | 224x224    | 83.9  | 24M     | 4.2G  |
| UniFormer-B+TL  | ImageNet-1K | 224x224    | 85.1  | 50M     | 8.3G  |
| UniFormer-L+TL  | ImageNet-1K | 224x224    | 85.6  | 100M    | 12.6G |
| UniFormer-S+TL  | ImageNet-1K | 384x384    | 84.6  | 22M     | 11.9G |
| UniFormer-S†+TL | ImageNet-1K | 384x384    | 84.9  | 24M     | 13.7G |
| UniFormer-B+TL  | ImageNet-1K | 384x384    | 86.0  | 50M     | 27.2G |
| UniFormer-L+TL  | ImageNet-1K | 384x384    | 86.3  | 100M    | 39.2G |

## Citation

```
@article{li2022uniformer,
    title={UniFormer: Unifying Convolution and Self-attention for Visual Recognition}, 
    author={Kunchang Li and Yali Wang and Junhao Zhang and Peng Gao and Guanglu Song and Yu Liu and Hongsheng Li and Yu Qiao},
    year={2022},
    eprint={2201.09450},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
@article{li2022uniformer,
    title={UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning}, 
    author={Kunchang Li and Yali Wang and Peng Gao and Guanglu Song and Yu Liu and Hongsheng Li and Yu Qiao},
    year={2022},
    eprint={2201.04676},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
