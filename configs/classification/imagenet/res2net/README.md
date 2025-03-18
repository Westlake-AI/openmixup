# Res2Net

> [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)

## Abstract

Representing features at multiple scales is of great importance for numerous vision tasks. Recent advances in backbone convolutional neural networks (CNNs) continually demonstrate stronger multi-scale representation ability, leading to consistent performance gains on a wide range of applications. However, most existing methods represent the multi-scale features in a layer-wise manner. In this paper, we propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer. The proposed Res2Net block can be plugged into the state-of-the-art backbone CNN models, e.g., ResNet, ResNeXt, and DLA. We evaluate the Res2Net block on all these models and demonstrate consistent performance gains over baseline models on widely-used datasets, e.g., CIFAR-100 and ImageNet. Further ablation studies and experimental results on representative computer vision tasks, i.e., object detection, class activation mapping, and salient object detection, further verify the superiority of the Res2Net over the state-of-the-art baseline methods.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/142573547-cde68abf-287b-46db-a848-5cffe3068faf.png" width="60%"/>
</div>

## Results and models

### ImageNet-1k

|        Model         | resolution | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |                               Config                               |                               Download                                |
| :------------------: | :--------: | :-------: | :------: | :-------: | :-------: | :----------------------------------------------------------------: | :-------------------------------------------------------------------: |
| Res2Net-50-14w-8s\*  |  224x224   |   25.06   |   4.22   |   78.14   |   93.85   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/res2net/res2net50_w14_s8_4xb64_ep100.py) | [model](https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth) |
| Res2Net-50-26w-8s\*  |  224x224   |   48.40   |   8.39   |   79.20   |   94.36   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/res2net/res2net50_w26_s8_4xb64_ep100.py) | [model](https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w26-s8_3rdparty_8xb32_in1k_20210927-f547a94b.pth) |
| Res2Net-101-26w-4s\* |  224x224   |   45.21   |   8.12   |   79.19   |   94.44   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/res2net/res2net101_w26_s4_4xb64_ep100.py) | [model](https://download.openmmlab.com/mmclassification/v0/res2net/res2net101-w26-s4_3rdparty_8xb32_in1k_20210927-870b6c36.pth) |

We follow the original training setting provided by the original paper. *Models with * are converted from the [official repo](https://github.com/Res2Net/Res2Net-PretrainedModels).* We don't ensure these config files' training accuracy.

## Citation

```
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
<<<<<<< HEAD
  journal={IEEE TPAMI},
=======
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
  year={2021},
  doi={10.1109/TPAMI.2019.2938758},
}
```
