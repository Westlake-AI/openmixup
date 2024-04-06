# StarNet

> [Rewrite the Stars](https://arxiv.org/abs/2403.19967)

## Abstract

Recent studies have drawn attention to the untapped potential of the "star operation" (element-wise multiplication) in network design. While intuitive explanations abound, the foundational rationale behind its application remains largely unexplored. Our study attempts to reveal the star operation's ability to map inputs into high-dimensional, non-linear feature spaces -- akin to kernel tricks -- without widening the network. We further introduce StarNet, a simple yet powerful prototype, demonstrating impressive performance and low latency under compact network structure and efficient budget. Like stars in the sky, the star operation appears unremarkable but holds a vast universe of potential. Our work encourages further exploration across tasks, with codes available at [this https URL](https://github.com/ma-xu/Rewrite-the-Stars/).

<div align=center>
<img src="https://github.com/Westlake-AI/openmixup/assets/44519745/49e026fc-780c-45f6-aedf-b8a2173015d7" width="95%"/>
</div>

## Results and models

### ImageNet-1k

| Model | Params(M) | Flops(G) | Top-1 (%) | Config |
|:---:|:---:|:---:|:---:|:---:|
| StarNet-S1\* | 2.87 | 0.43 | 73.50 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/starnet/starnet_s1_8xb256_ep300.py) |
| StarNet-S2\* | 3.68 | 0.55 | 74.80 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/starnet/starnet_s2_8xb256_ep300.py) |
| StarNet-S3\* | 5.75 | 0.77 | 77.30 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/starnet/starnet_s3_8xb256_ep300.py) |
| StarNet-S4\* | 7.48 | 1.07 | 78.40 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/starnet/starnet_s4_8xb256_ep300.py) |

We follow the original training setting provided by the original paper. *Models with * are converted from the [official repo](https://github.com/ma-xu/Rewrite-the-Stars).* We don't ensure these config files' training accuracy.

## Citation

```
@inproceedings{ma2024rewrite,
    title={Rewrite the Stars},
    author={Xu Ma and Xiyang Dai and Yue Bai and Yizhou Wang and Yun Fu},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```
