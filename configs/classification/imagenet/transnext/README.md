# TransNeXt

> [TransNeXt: Robust Foveal Visual Perception for Vision Transformers](https://arxiv.org/abs/2311.17132)

## Abstract

Due to the depth degradation effect in residual connections, many efficient Vision Transformers models that rely on stacking layers for information exchange often fail to form sufficient information mixing, leading to unnatural visual perception. To address this issue, in this paper, we propose **Aggregated Attention**, a biomimetic design-based token mixer that simulates biological foveal vision and continuous eye movement while enabling each token on the feature map to have a global perception. Furthermore, we incorporate learnable tokens that interact with conventional queries and keys, which further diversifies the generation of affinity matrices beyond merely relying on the similarity between queries and keys. Our approach does not rely on stacking for information exchange, thus effectively avoiding depth degradation and achieving natural visual perception. Additionally, we propose **Convolutional GLU**, a channel mixer that bridges the gap between GLU and SE mechanism, which empowers each token to have channel attention based on its nearest neighbor image features, enhancing local modeling capability and model robustness. We combine aggregated attention and convolutional GLU to create a new visual backbone called **TransNeXt**. Extensive experiments demonstrate that our TransNeXt achieves state-of-the-art performance across multiple model sizes. At a resolution of $224^2$, TransNeXt-Tiny attains an ImageNet accuracy of **84.0\%**, surpassing ConvNeXt-B with **69\%** fewer parameters. Our TransNeXt-Base achieves an ImageNet accuracy of **86.2\%** and an ImageNet-A accuracy of **61.6\%** at a resolution of $384^2$, a COCO object detection mAP of **57.1**, and an ADE20K semantic segmentation mIoU of **54.7**.

<div align=center>
<img src="https://github.com/Westlake-AI/openmixup/assets/44519745/49e026fc-780c-45f6-aedf-b8a2173015d7" width="95%"/>
</div>

## Results and models

### ImageNet-1k

| Model | Params(M) | Flops(G) | Top-1 (%) | Config |
|:---:|:---:|:---:|:---:|:---:|
| TransNeXt-Micro\* | 12.8 | 2.7 | 82.50 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/transnext/transnext_micro_8xb128_ep300.py) |
| TransNeXt-Tiny\* | 28.2 | 5.7 | 84.00 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/transnext/transnext_tiny_8xb128_ep300.py) |
| TransNeXt-Small\* | 49.7 | 10.3 | 94.70 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/transnext/transnext_small_8xb128_ep300.py) |
| TransNeXt-Base\* | 89.7 | 18.4 | 84.80 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/transnext/transnext_base_8xb128_ep300.py) |

We follow the original training setting provided by the original paper. *Models with * are converted from the [official repo](https://github.com/DaiShiResearch/TransNeXt).* We don't ensure these config files' training accuracy.

## Citation

```
@misc{shi2023transnext,
  author = {Dai Shi},
  title = {TransNeXt: Robust Foveal Visual Perception for Vision Transformers},
  year = {2023},
  eprint = {arXiv:2311.17132},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
