# Fast Vision Transformers with HiLo Attention

> [Fast Vision Transformers with HiLo Attention](https://arxiv.org/abs/2205.13213)

## Abstract

Vision Transformers (ViTs) have triggered the most recent and significant breakthroughs in computer vision. Their efficient designs are mostly guided by the indirect metric of computational complexity, i.e., FLOPs, which however has a clear gap with the direct metric such as throughput. Thus, we propose to use the direct speed evaluation on the target platform as the design principle for efficient ViTs. Particularly, we introduce LITv2, a simple and effective ViT which performs favourably against the existing state-of-the-art methods across a spectrum of different model sizes with faster speed. At the core of LITv2 is a novel self-attention mechanism, which we dub HiLo. HiLo is inspired by the insight that high frequencies in an image capture local fine details and low frequencies focus on global structures, whereas a multi-head self-attention layer neglects the characteristic of different frequencies. Therefore, we propose to disentangle the high/low frequency patterns in an attention layer by separating the heads into two groups, where one group encodes high frequencies via self-attention within each local window, and another group performs the attention to model the global relationship between the average-pooled low-frequency keys from each window and each query position in the input feature map. Benefit from the efficient design for both groups, we show that HiLo is superior to the existing attention mechanisms by comprehensively benchmarking on FLOPs, speed and memory consumption on GPUs. Powered by HiLo, LITv2 serves as a strong backbone for mainstream vision tasks including image classification, dense detection and segmentation. Code is available at this https [URL](https://github.com/ziplab/LITv2).

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/178601915-18e96064-4698-4d45-b55b-e592b33fb593.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

|   Model   |   Pretrain   | resolution | Params(M) | Flops(G) | Throughput (imgs/s) | Top-1 (%) |                               Config                                |                               Download                                |
| :-------: | :----------: | :--------: | :-------: | :------: | :-------: | :-------: | :-----------------------------------------------------------------: | :-------------------------------------------------------------------: |
|  LITv2-S  | From scratch |  224x224   |     28    |    3.7   | 1,471               |    81.7   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/lit_v2/lit_v2_small_8xb128_cos_fp16_ep300.py) | model / log |
| LITv2-S\* | From scratch |  224x224   |     28    |    3.7   | 1,471               |    82.0   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/lit_v2/lit_v2_small_8xb128_cos_fp16_ep300.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_s.pth) / [log](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_m_log.txt) |
| LITv2-M\* | From scratch |  224x224   |     49    |    7.5   | 812                 |    83.3   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/lit_v2/lit_v2_medium_8xb128_cos_fp16_ep300.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_m.pth) / [log](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_m_log.txt) |
| LITv2-B\* | From scratch |  224x224   |     87    |   13.2   | 602                 |    84.7   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/lit_v2/lit_v2_base_8xb128_cos_fp16_ep300.py) | [model](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_b.pth) / [log](https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_b_log.txt) |


We follow the original training setting provided by the [official repo](https://github.com/ziplab/LITv2) and throughput is averaged over 30 runs. *Note that models with \* are converted from the [official repo](https://github.com/ziplab/LITv2).* We reproduce LITv2-S training 300 epochs.

## Citation

```
@article{pan2022hilo
  title={Fast Vision Transformers with HiLo Attention},
  author={Pan, Zizheng and Cai, Jianfei and Zhuang, Bohan},
  journal={arXiv preprint arXiv:2205.13213},
  year={2022}
}
```
