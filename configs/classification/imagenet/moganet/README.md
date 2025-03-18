# Efficient Multi-order Gated Aggregation Network

> [Efficient Multi-order Gated Aggregation Network](https://arxiv.org/abs/2211.03295)

## Abstract

Since the recent success of Vision Transformers (ViTs), explorations toward ViT-style architectures have triggered the resurgence of ConvNets. In this work, we explore the representation ability of modern ConvNets from a novel view of multi-order game-theoretic interaction, which reflects inter-variable interaction effects w.r.t.~contexts of different scales based on game theory. Within the modern ConvNet framework, we tailor the two feature mixers with conceptually simple yet effective depthwise convolutions to facilitate middle-order information across spatial and channel spaces respectively. In this light, a new family of pure ConvNet architecture, dubbed MogaNet, is proposed, which shows excellent scalability and attains competitive results among state-of-the-art models with more efficient use of parameters on ImageNet and multifarious typical vision benchmarks, including COCO object detection, ADE20K semantic segmentation, 2D\&3D human pose estimation, and video prediction. Typically, MogaNet hits 80.0\% and 87.8\% top-1 accuracy with 5.2M and 181M parameters on ImageNet, outperforming ParC-Net-S and ConvNeXt-L while saving 59\% FLOPs and 17M parameters. The source code is available at https://github.com/Westlake-AI/MogaNet.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/200625735-86bd2237-5bbe-43c1-ab37-049810b8d8a1.jpg" width="100%"/>
</div>

## Results and models

Here, we provide ImageNet classification results and pre-trained models. You can download all files from [openmixup-moganet-in1k-weights](https://github.com/Westlake-AI/openmixup/releases/tag/moganet-in1k-weights), [timm-moganet-in1k-weights](https://github.com/Westlake-AI/MogaNet/releases/tag/moganet-in1k-weights), or **Baidu Cloud**: [MogaNet (z8mf)](https://pan.baidu.com/s/1d5MTTC66gegehmfZvCQRUA?pwd=z8mf). Please refer to [MogaNet](https://github.com/Westlake-AI/MogaNet) for replementations of classification and full implementations of dense prediction tasks (COCO object detection and ADE20K semantic segmentation).

### Usage

* Demo: A simple Google Colab [demo](https://github.com/Westlake-AI/MogaNet/demo.ipynb) of MogaNet which run the steps to perform inference for image classification.
* Analysis tools: In OpenMixup, use [vis_cam.py](https://github.com/Westlake-AI/openmixup/tools/visualizations/vis_cam.py) and [get_flops.py](https://github.com/Westlake-AI/openmixup/tools/analysis_tools/get_flops.py) to visualize Grad-CAM activation maps and caculate FLOPs. In [MogaNet](https://github.com/Westlake-AI/MogaNet), the analysis can be conducted by [cam_image.py](https://github.com/Westlake-AI/MogaNet/cam_image.py) and [get_flops.py](https://github.com/Westlake-AI/MogaNet/get_flops.py).
* Warning of `attn_force_fp32`: During fp16 training, we force to run the gating functions with fp32 to avoid inf or nan. We found that if we use `attn_force_fp32=True` during training, it should also keep `attn_force_fp32=True` during evaluation. This might be caused by the difference between the output results of using `attn_force_fp32` or not. It will not affect performances of fully fine-tuning but the results of transfer learning (e.g., COCO Mask-RCNN freezes the parameters of the first stage). We set it to true by default in OpenMixup while removing it in [MogaNet](https://github.com/Westlake-AI/MogaNet) implementation. For example, you can use [moga_small_ema_sz224_8xb128_ep300](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_small_ema_sz224_8xb128_ep300.pth) with `attn_force_fp32=True` while using [moga_small_ema_sz224_8xb128_no_forcefp32_ep300](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_small_ema_sz224_8xb128_no_forcefp32_ep300.pth) with `attn_force_fp32=False`.

### ImageNet-1k

| Model | Pretrain | Setting | resolution | Params(M) | Flops(G) | Top-1 (%) | Config | Download |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|:---:|
| MogaNet-XT | From scratch | DeiT | 224x224 | 2.97 | 0.80 | 76.5 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_xtiny_sz224_8xb128_fp16_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_xtiny_sz224_8xb128_fp16_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_xtiny_sz224_8xb128_fp16_ep300.log.json) |
| MogaNet-XT | From scratch | DeiT | 256x256 | 2.97 | 1.04 | 77.2 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_xtiny_sz256_8xb128_fp16_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_xtiny_sz256_8xb128_fp16_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_xtiny_sz256_8xb128_fp16_ep300.log.json) |
| MogaNet-XT\* | From scratch | DeiT | 256x256 | 2.97 | 1.04 | 77.6 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_xtiny_deit3_sz256_lr2e_3_8xb128_fp16_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_xtiny_deit3_sz256_lr2e_3_8xb128_fp16_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_xtiny_deit3_sz256_lr2e_3_8xb128_fp16_ep300.log.json) |
| MogaNet-T | From scratch | DeiT | 224x224 | 5.20 | 1.10 | 79.0 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_tiny_sz224_8xb128_fp16_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_tiny_sz224_8xb128_fp16_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_tiny_sz224_8xb128_fp16_ep300.log.json) |
| MogaNet-T | From scratch | DeiT | 256x256 | 5.20 | 1.44 | 79.6 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_tiny_sz256_8xb128_fp16_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_tiny_sz256_8xb128_fp16_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_tiny_sz256_8xb128_fp16_ep300.log.json) |
| MogaNet-T\* | From scratch | DeiT | 256x256 | 5.20 | 1.44 | 80.0 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_tiny_deit3_sz256_lr2e_3_8xb128_fp16_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_tiny_deit3_sz256_lr2e_3_8xb128_fp16_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_tiny_deit3_sz256_lr2e_3_8xb128_fp16_ep300.log.json) |
| MogaNet-S | From scratch | DeiT | 224x224 | 25.3 | 4.97 | 83.4 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_small_ema_sz224_8xb128_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_small_ema_sz224_8xb128_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_small_ema_sz224_8xb128_ep300.log.json) |
| MogaNet-B | From scratch | DeiT | 224x224 | 43.9 | 9.93 | 84.3 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_base_ema_sz224_8xb128_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_base_ema_sz224_8xb128_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_base_ema_sz224_8xb128_ep300.log.json) |
| MogaNet-L | From scratch | DeiT | 224x224 | 82.5 | 15.9 | 84.7 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_large_ema_sz224_8xb64_accu2_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_large_ema_sz224_8xb64_accu2_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_large_ema_sz224_8xb64_accu2_ep300.log.json) |
| MogaNet-XL | From scratch | DeiT | 224x224 | 180.8 | 34.5 | 85.1 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_xlarge_ema_sz224_8xb32_accu2_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_xlarge_ema_sz224_8xb32_accu2_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_xlarge_ema_sz224_8xb32_accu2_ep300.log.json) |
| MogaNet-XT | From scratch | RSB A3 | 160x160 | 2.97 | 0.80 | 72.8 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_xtiny_rsb_a3_sz160_8xb256_ep100.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_xtiny_sz160_rsb_a3_8xb256_ep100.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_xtiny_sz160_rsb_a3_8xb256_ep100.log.json) |
| MogaNet-T | From scratch | RSB A3 | 160x160 | 5.20 | 1.10 | 75.4 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_tiny_rsb_a3_sz160_8xb256_ep100.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_tiny_sz160_rsb_a3_8xb256_ep100.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_tiny_sz160_rsb_a3_8xb256_ep100.log.json) |
| MogaNet-S | From scratch | RSB A3 | 160x160 | 25.3 | 4.97 | 81.1 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_small_rsb_a3_sz160_8xb256_ep100.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_small_sz160_rsb_a3_8xb256_ep100.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_small_sz160_rsb_a3_8xb256_ep100.log.json) |
| MogaNet-B | From scratch | RSB A3 | 160x160 | 43.9 | 9.93 | 82.2 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_base_rsb_a3_sz160_8xb128_accu2_ep100.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_base_rsb_a3_sz160_8xb128_accu2_ep100.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_base_rsb_a3_sz160_8xb128_accu2_ep100.log.json) |
| MogaNet-L | From scratch | RSB A3 | 160x160 | 43.9 | 9.93 | 83.2 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/moganet/moga_large_rsb_a3_sz160_8xb128_accu2_ep100.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_large_rsb_a3_sz160_8xb128_accu2_ep100.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/moganet-in1k-weights/moga_large_rsb_a3_sz160_8xb128_accu2_ep100.log.json) |

We provide the config files according to the original training setting described in the [paper](https://arxiv.org/abs/2211.03295) and report FLOPs with the test resolutions (224x224 or 256x256). Note that \* denotes the refined training setting of lightweight models with [3-Augment](https://arxiv.org/abs/2204.07118). We can get a slightly better performance with [Precise BN](https://arxiv.org/abs/2105.07576) for all MogaNet variants.

## Citation

```
<<<<<<< HEAD
@article{Li2022MogaNet,
  title={Efficient Multi-order Gated Aggregation Network},
  author={Siyuan Li and Zedong Wang and Zicheng Liu and Cheng Tan and Haitao Lin and Di Wu and Zhiyuan Chen and Jiangbin Zheng and Stan Z. Li},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.03295}
=======
@inproceedings{iclr2024MogaNet,
  title={Efficient Multi-order Gated Aggregation Network},
  author={Siyuan Li and Zedong Wang and Zicheng Liu and Cheng Tan and Haitao Lin and Di Wu and Zhiyuan Chen and Jiangbin Zheng and Stan Z. Li},
  booktitle={International Conference on Learning Representations},
  year={2024}
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
}
```
