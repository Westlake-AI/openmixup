# UniFormer: Unifying Convolution and Self-attention for Visual Recognition

> [UniFormer: Unifying Convolution and Self-attention for Visual Recognition](https://arxiv.org/abs/2201.09450)

## Abstract

It is a challenging task to learn discriminative representation from images and videos, due to large local redundancy and complex global dependency in these visual data. Convolution neural networks (CNNs) and vision transformers (ViTs) have been two dominant frameworks in the past few years. Though CNNs can efficiently decrease local redundancy by convolution within a small neighborhood, the limited receptive field makes it hard to capture global dependency. Alternatively, ViTs can effectively capture long-range dependency via self-attention, while blind similarity comparisons among all the tokens lead to high redundancy. To resolve these problems, we propose a novel Unified transFormer (UniFormer), which can seamlessly integrate the merits of convolution and self-attention in a concise transformer format. Different from the typical transformer blocks, the relation aggregators in our UniFormer block are equipped with local and global token affinity respectively in shallow and deep layers, allowing to tackle both redundancy and dependency for efficient and effective representation learning. Finally, we flexibly stack our UniFormer blocks into a new powerful backbone, and adopt it for various vision tasks from image to video domain, from classification to dense prediction. Without any extra training data, our UniFormer achieves 86.3 top-1 accuracy on ImageNet-1K classification. With only ImageNet-1K pre-training, it can simply achieve state-of-the-art performance in a broad range of downstream tasks, e.g., it obtains 82.9/84.8 top-1 accuracy on Kinetics-400/600, 60.9/71.2 top-1 accuracy on Something-Something V1/V2 video classification tasks, 53.8 box AP and 46.4 mask AP on COCO object detection task, 50.8 mIoU on ADE20K semantic segmentation task, and 77.4 AP on COCO pose estimation task. Code is available at [this https URL](https://github.com/Sense-X/UniFormer).

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/174448564-fc87501f-8d75-4b6c-9545-f1c6cd4043a7.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

|       Model       |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) |                               Config                                |                               Download                                |
| :---------------: | :----------: | :--------: | :-------: | :------: | :-------: | :-----------------------------------------------------------------: | :-------------------------------------------------------------------: |
| UniFormer-T       | From scratch |  224x224   |   5.55    |   0.88   |   78.02   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/uniformer/uniformer_tiny_8xb128_ep300.py) | [log](https://github.com/Westlake-AI/openmixup/releases/download/open-in1k-weights/uniformer_tiny_8xb128_ep300.log.json) |
| UniFormer-S       | From scratch |  224x224   |   21.5    |   3.44   |   82.29   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/uniformer/uniformer_small_8xb128_fp16_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/open-in1k-weights/uniformer_small_8xb128_fp16_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/open-in1k-weights/uniformer_small_8xb128_fp16_ep300.log.json) |
| UniFormer-S\*     | From scratch |  224x224   |   21.5    |   3.44   |   82.90   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/uniformer/uniformer_small_8xb128_fp16_ep300.py) | [model](https://drive.google.com/file/d/1-uepH3Q3BhTmWU6HK-sGAGQC_MpfIiPD/view?usp=sharing) | [log](https://drive.google.com/file/d/10ThKb9YOpCiHW8HL10dRuZ0lQSPJidO7/view?usp=sharing) |
| UniFormer-S†\*    | From scratch |  224x224   |   24.0    |   4.21   |   83.40   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/uniformer/uniformer_small_plus_8xb128_ep300.py) | [model](https://drive.google.com/file/d/1-uepH3Q3BhTmWU6HK-sGAGQC_MpfIiPD/view?usp=sharing) | [log](https://drive.google.com/file/d/10ThKb9YOpCiHW8HL10dRuZ0lQSPJidO7/view?usp=sharing) |
| UniFormer-B\*     | From scratch |  224x224   |   49.8    |   8.27   |   83.90   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/uniformer/uniformer_base_8xb128_ep300.py) | [model](https://drive.google.com/file/d/1-uepH3Q3BhTmWU6HK-sGAGQC_MpfIiPD/view?usp=sharing) | [log](https://drive.google.com/file/d/10ThKb9YOpCiHW8HL10dRuZ0lQSPJidO7/view?usp=sharing) |
| UniFormer-S+TL\*  | From scratch |  224x224   |   21.5    |   3.44   |   83.40   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/uniformer/uniformer_small_8xb128_fp16_ep300.py) | [model](https://drive.google.com/file/d/1-uepH3Q3BhTmWU6HK-sGAGQC_MpfIiPD/view?usp=sharing) | [log](https://drive.google.com/file/d/10ThKb9YOpCiHW8HL10dRuZ0lQSPJidO7/view?usp=sharing) |
| UniFormer-B+TL\*  | From scratch |  224x224   |   49.8    |   8.27   |   85.10   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/uniformer/uniformer_base_8xb128_ep300.py) | [model](https://drive.google.com/file/d/1-uepH3Q3BhTmWU6HK-sGAGQC_MpfIiPD/view?usp=sharing) | [log](https://drive.google.com/file/d/10ThKb9YOpCiHW8HL10dRuZ0lQSPJidO7/view?usp=sharing) |

We follow the original training setting provided by the [official repo](https://github.com/Sense-X/UniFormer). *Note that models with \* are converted from [the official repo](https://github.com/Sense-X/UniFormer/blob/main/image_classification), † denotes using `ConvStem` in UniFormer. TL denotes training the model with [Token Labeling](https://arxiv.org/abs/2104.10858) as [LV-ViT](https://github.com/zihangJiang/TokenLabeling).* We reproduce the performances of UniFormer-T and UniFormer-S training 300 epochs, and UniFormer-T is designed according to VAN-T.

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
