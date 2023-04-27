# Context Clusters

> [Image as Set of Points](https://arxiv.org/abs/2303.01494)

## Abstract

What is an image and how to extract latent features? Convolutional Networks (ConvNets) consider an image as organized pixels in a rectangular shape and extract features via convolutional operation in local region; Vision Transformers (ViTs) treat an image as a sequence of patches and extract features via attention mechanism in a global range. In this work, we introduce a straightforward and promising paradigm for visual representation, which is called Context Clusters. Context clusters (CoCs) view an image as a set of unorganized points and extract features via simplified clustering algorithm. In detail, each point includes the raw feature (e.g., color) and positional information (e.g., coordinates), and a simplified clustering algorithm is employed to group and extract deep features hierarchically. Our CoCs are convolution- and attention-free, and only rely on clustering algorithm for spatial interaction. Owing to the simple design, we show CoCs endow gratifying interpretability via the visualization of clustering process. Our CoCs aim at providing a new perspective on image and visual representation, which may enjoy broad applications in different domains and exhibit profound insights. Even though we are not targeting SOTA performance, COCs still achieve comparable or even better results than ConvNets or ViTs on several benchmarks. Codes are available at: this https URL. 

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/232592809-ad1005a3-1b99-4eb5-8ffb-3094af84ae80.png" width="90%"/>
</div>

## Results and models

This page is based on the [official repo](https://github.com/ma-xu/Context-Cluster).

### ImageNet-1k

| Model | Params(M) | Flops(G) | Top-1 (%) | Throughputs | Config | Download |
| :---: | :-------: | :------: | :-------: | :---------: | :----: | :------: |
| ContextCluster-tiny\* | 5.6 | 1.10 | 71.8 | 518.4 | [config](coc_tiny_8xb256_ep300.py) | [model](https://drive.google.com/drive/folders/1Q_6W3xKMX63aQOBaqiwX5y1fCj4hVOIA?usp=sharing) |
| ContextCluster-tiny_plain\* (w/o region partition) | 5.6 | 1.10 | 72.9 |  -  | [config](coc_tiny_plain_8xb256_ep300.py) | [model](https://web.northeastern.edu/smilelab/xuma/ContextCluster/checkpoints/coc_tiny_plain/coc_tiny_plain.pth.tar) |
| ContextCluster-small\* | 14.7 | 2.78 | 77.5 | 513.0 | [config](coc_small_8xb256_ep300.py) | [model](https://drive.google.com/drive/folders/1WSmnbSgy1I1HOTTTAQgOKEzXSvd3Kmh-?usp=sharing) |
| ContextCluster-medium\* | 29.3 | 5.90 | 81.0 | 325.2 | [config](coc_medium_8xb256_ep300.py) | [model](https://drive.google.com/drive/folders/1sPxnEHb2AHDD9bCQh6MA0I_-7EBrvlT5?usp=sharing) |
| ContextCluster-tiny | 5.6 | 1.10 | 72.7 | 518.4 | [config](coc_tiny_8xb256_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/open-in1k-weights/coc_tiny_8xb256_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/open-in1k-weights/) |
| ContextCluster-tiny_plain (w/o region partition) | 5.6 | 1.10 | 73.2 |  -  | [config](coc_tiny_plain_8xb256_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/open-in1k-weights/coc_tiny_plain_8xb256_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/open-in1k-weights/coc_tiny_plain_8xb256_ep300.log.json) |
| ContextCluster-small | 14.7 | 2.78 | 77.7 | 513.0 | [config](coc_small_8xb256_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/open-in1k-weights/coc_small_8xb256_ep300.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/open-in1k-weights/coc_small_8xb256_ep300.log.json) |

We follow the original training setting provided by the [official repo](https://github.com/ma-xu/Context-Cluster) to reproduce better performance of ContextCluster variants. *Models with * are converted from the [official repo](https://github.com/ma-xu/Context-Cluster).*

## Citation

```bibtex
@inproceedings{iclr2023coc,
      title={Image as Set of Points},
      author={Xu Ma and Yuqian Zhou and Huan Wang and Can Qin and Bin Sun and Chang Liu and Yun Fu},
      booktitle={The Eleventh International Conference on Learning Representations},
      year={2023},
      url={https://openreview.net/forum?id=awnvqZja69}
}
```
