# UniRepLKNet

> [UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition](https://arxiv.org/abs/2311.15599)

## Abstract

Large-kernel convolutional neural networks (ConvNets) have recently received extensive research attention, but two unresolved and critical issues demand further investigation. 1) The architectures of existing large-kernel ConvNets largely follow the design principles of conventional ConvNets or transformers, while the architectural design for large-kernel ConvNets remains under-addressed. 2) As transformers have dominated multiple modalities, it remains to be investigated whether ConvNets also have a strong universal perception ability in domains beyond vision. In this paper, we contribute from two aspects. 1) We propose four architectural guidelines for designing large-kernel ConvNets, the core of which is to exploit the essential characteristics of large kernels that distinguish them from small kernels - they can see wide without going deep. Following such guidelines, our proposed large-kernel ConvNet shows leading performance in image recognition (ImageNet accuracy of 88.0%, ADE20K mIoU of 55.6%, and COCO box AP of 56.4%), demonstrating better performance and higher speed than the recent powerful competitors. 2) We discover large kernels are the key to unlocking the exceptional performance of ConvNets in domains where they were originally not proficient. With certain modality-related preprocessing approaches, the proposed model achieves state-of-the-art performance on time-series forecasting and audio recognition tasks even without modality-specific customization to the architecture. All the code and models are publicly available on GitHub and Huggingface. 

<div align=center>
<img src="https://github.com/Westlake-AI/openmixup/assets/44519745/7004cbb6-d56f-4eed-b666-9c49c151f7cf" width="80%"/>
</div>

## Results and models

### ImageNet-1k

| Model | Resolution | Top-1 (%) | Params(M) | Flops(G) | Config | Weights |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| UniRepLKNet-A | 224x224 | 77.0 | 4.4M  | 0.6G | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/unireplknet/unireplknet_a_8xb128_accu4_ep300.py) | [ckpt](https://drive.google.com/file/d/1jUB-lq6NMTbeBvGTDvAarKWh-ZfMMZWt/view?usp=drive_link) |
| UniRepLKNet-F | 224x224 | 78.6 | 6.2M  | 0.9G | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/unireplknet/unireplknet_f_8xb128_accu4_ep300.py) | [ckpt](https://drive.google.com/file/d/1vYqhCNx3q-z0fVT4UZecFTUmb9IDaYh9/view?usp=drive_link) |
| UniRepLKNet-P | 224x224 | 80.2 | 10.7M  | 1.6G | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/unireplknet/unireplknet_p_8xb128_accu4_ep300.py) | [ckpt](https://drive.google.com/file/d/1D7rljWnnzEGDn8MDkvAWJ8qd1SCix6Vm/view?usp=drive_link) |
| UniRepLKNet-N | 224x224 | 81.6 | 18.3M | 2.8G | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/unireplknet/unireplknet_n_8xb128_accu4_ep300.py) | [ckpt](https://drive.google.com/file/d/1tMHOl55C7h44ag8SLUuaP0bBUUpVXhKj/view?usp=drive_link) |
| UniRepLKNet-T | 224x224 | 83.2 | 31M | 4.9G | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/unireplknet/unireplknet_t_8xb128_accu4_ep300.py) | [ckpt](https://drive.google.com/file/d/12Xon3FWkzEQV1nnNsF2U8XDMD-7NO2cJ/view?usp=drive_link) |
| UniRepLKNet-S | 224x224 | 83.9 | 56M   | 9.1G | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/unireplknet/unireplknet_s_8xb128_accu4_ep300.py) | [ckpt](https://drive.google.com/file/d/11YEOcKs4WNprRzCvKe-fB5z-l7zQv3kb/view?usp=drive_link) |

We follow the original training setting provided by the [original paper](https://arxiv.org/abs/2311.15599). *Note that models with \* are converted from the [official repo](https://github.com/AILab-CVC/UniRepLKNet/tree/main/Image).*

## Citation

```
@article{ding2023unireplknet,
  title={UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition},
  author={Ding, Xiaohan and Zhang, Yiyuan and Ge, Yixiao and Zhao, Sijie and Song, Lin and Yue, Xiangyu and Shan, Ying},
  journal={arXiv preprint arXiv:2311.15599},
  year={2023}
}
```
