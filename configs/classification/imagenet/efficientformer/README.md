# EfficientFormer

> [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

## Abstract

Vision Transformers (ViT) have shown rapid progress in computer vision tasks, achieving promising results on various benchmarks. However, due to the massive number of parameters and model design, e.g., attention mechanism, ViT-based models are generally times slower than lightweight convolutional networks. Therefore, the deployment of ViT for real-time applications is particularly challenging, especially on resource-constrained hardware such as mobile devices. Recent efforts try to reduce the computation complexity of ViT through network architecture search or hybrid design with MobileNet block, yet the inference speed is still unsatisfactory. This leads to an important question: can transformers run as fast as MobileNet while obtaining high performance? To answer this, we first revisit the network architecture and operators used in ViT-based models and identify inefficient designs. Then we introduce a dimension-consistent pure transformer (without MobileNet blocks) as a design paradigm.  Finally, we perform latency-driven slimming to get a series of final models dubbed EfficientFormer. Extensive experiments show the superiority of EfficientFormer in performance and speed on mobile devices. Our fastest model, EfficientFormer-L1, achieves 79.2% top-1 accuracy on ImageNet-1K with only 1.6 ms inference latency on iPhone 12 (compiled with CoreML), which runs as fast as MobileNetV2×1.4 (1.6 ms, 74.7% top-1), and our largest model, EfficientFormer-L7, obtains 83.3% accuracy with only 7.0 ms latency. Our work proves that properly designed transformers can reach extremely low latency on mobile devices while maintaining high performance.

<div align=center>
<img src="https://user-images.githubusercontent.com/18586273/180713426-9d3d77e3-3584-42d8-9098-625b4170d796.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

|        Model         |   Pretrain   | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                                 Config                                  |                                  Download                                  |
| :------------------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :---------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| EfficientFormer-l1\* | From scratch |  224x224  |   12.19   |   1.30   |   80.46   |   94.99   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/efficientformer/efficientformer_l1_8xb256_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientformer/efficientformer-l1_3rdparty_in1k_20220803-d66e61df.pth) |
| EfficientFormer-l3\* | From scratch |  224x224  |   31.41   |   3.93   |   82.45   |   96.18   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/efficientformer/efficientformer_l3_8xb256_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientformer/efficientformer-l3_3rdparty_in1k_20220803-dde1c8c5.pth) |
| EfficientFormer-l7\* | From scratch |  224x224  |   82.23   |  10.16   |   83.40   |   96.60   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/efficientformer/efficientformer_l7_8xb256_ep300.py) | [model](https://download.openmmlab.com/mmclassification/v0/efficientformer/efficientformer-l7_3rdparty_in1k_20220803-41a552bb.pth) |

We follow the original training setting provided by the [official repo](https://github.com/snap-research/EfficientFormer) and the [original paper](https://arxiv.org/abs/2206.01191). *However, this repo does not support the distillation loss in EfficientFormer, where we use the normal classification head instread. Note that models with \* are converted from the [official repo](https://github.com/snap-research/EfficientFormer).*

## Citation

```
@misc{2022efficientformer,
  author = {Li, Yanyu and Yuan, Geng and Wen, Yang and Hu, Eric and Evangelidis, Georgios and Tulyakov, Sergey and Wang, Yanzhi and Ren, Jian},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {EfficientFormer: Vision Transformers at MobileNet Speed},
  publisher = {arXiv},
  year = {2022},
  doi = {10.48550/ARXIV.2206.01191},
  url = {https://arxiv.org/abs/2206.01191},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
