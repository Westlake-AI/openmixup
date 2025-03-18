# ConvNeXt V2

> [Co-designing and Scaling ConvNets with Masked Autoencoders](http://arxiv.org/abs/2301.00808)

## Abstract

Driven by improved architectures and better representation learning frameworks, the field of visual recognition has enjoyed rapid modernization and performance boost in the early 2020s. For example, modern ConvNets, represented by ConvNeXt, have demonstrated strong performance in various scenarios. While these models were originally designed for supervised learning with ImageNet labels, they can also potentially benefit from self-supervised learning techniques such as masked autoencoders (MAE). However, we found that simply combining these two approaches leads to subpar performance. In this paper, we propose a fully convolutional masked autoencoder framework and a new Global Response Normalization (GRN) layer that can be added to the ConvNeXt architecture to enhance inter-channel feature competition. This co-design of self-supervised learning techniques and architectural improvement results in a new model family called ConvNeXt V2, which significantly improves the performance of pure ConvNets on various recognition benchmarks, including ImageNet classification, COCO detection, and ADE20K segmentation. We also provide pre-trained ConvNeXt V2 models of various sizes, ranging from an efficient 3.7M-parameter Atto model with 76.7% top-1 accuracy on ImageNet, to a 650M Huge model that achieves a state-of-the-art 88.9% accuracy using only public training data.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/210496285-f235083f-218f-4153-8e21-c8a64481a2f5.png" width="60%"/>
</div>

## Results and models

This page is based on documents in [MMPretrain](https://github.com/open-mmlab/mmpretrain).

### ImageNet-1k

|      Model     |   Pretrain   | Params(M)  | Flops(G)  | Top-1 (%) | Top-5 (%) | Config | Download |
| :------------: | :----------: | :--------: | :-------: | :-------: | :-------: | :----: | :------: |
| ConvNeXt-V2-Atto  |   FCMAE   |    3.71    |   0.55    |   76.64   |   93.04   | [config](convnext_v2_atto_8xb128_fp16_ep600.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-atto_fcmae-pre_3rdparty_in1k_20230104-23765f83.pth) |
| ConvNeXt-V2-Femto |   FCMAE   |    5.23    |   0.78    |   78.48   |   93.98   | [config](convnext_v2_femto_8xb128_fp16_ep600.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-femto_fcmae-pre_3rdparty_in1k_20230104-92a75d75.pth) |
| ConvNeXt-V2-Pico  |   FCMAE   |    9.07    |   1.37    |   80.31   |   95.08   | [config](convnext_v2_pico_8xb128_fp16_ep600.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-pico_fcmae-pre_3rdparty_in1k_20230104-d20263ca.pth) |
| ConvNeXt-V2-Nano  |   FCMAE   |   15.62    |   2.45    |   81.86   |   95.75   | [config](convnext_v2_nano_8xb128_fp16_ep600.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-nano_fcmae-pre_3rdparty_in1k_20230104-fe1aaaf2.pth) |
| ConvNeXt-V2-Tiny  |   FCMAE   |   28.64    |   4.47    |   82.94   |   96.29   | [config](convnext_v2_tiny_8xb128_fp16_ep600.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-tiny_fcmae-pre_3rdparty_in1k_20230104-471a86de.pth) |
| ConvNeXt-V2-Base  |   FCMAE   |   88.72    |   15.38   |   84.87   |   97.08   | [config](convnext_v2_base_8xb128_fp16_ep600.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-pre_3rdparty_in1k_20230104-00a70fa4.pth) |
| ConvNeXt-V2-Large |   FCMAE   |   197.96   |   34.40   |   85.76   |   97.59   | [config](convnext_v2_large_8xb128_fp16_ep600.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-large_fcmae-pre_3rdparty_in1k_20230104-ef393013.pth) |
| ConvNeXt-V2-Huge  |   FCMAE   |   660.29   |  115.00   |   86.25   |   97.75   | [config](convnext_v2_huge_8xb128_fp16_ep600.py) | [model](https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-huge_fcmae-pre_3rdparty_in1k_20230104-f795e5b8.pth) |

*Models with * are converted from the [official repo](https://github.com/facebookresearch/ConvNeXt-V2).*

## Citation

```bibtex
<<<<<<< HEAD
@article{Woo2023ConvNeXtV2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon and Saining Xie},
  year={2023},
  journal={arXiv preprint arXiv:2301.00808},
=======
@inproceedings{Woo2023ConvNeXtV2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Sanghyun Woo and Shoubhik Debnath and Ronghang Hu and Xinlei Chen and Zhuang Liu and In-So Kweon and Saining Xie},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
}
```
