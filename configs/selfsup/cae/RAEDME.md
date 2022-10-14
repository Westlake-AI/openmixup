# CAE

> [Context Autoencoder for Self-Supervised Representation Learning](https://arxiv.org/abs/2202.03026)

## Abstract

We present a novel masked image modeling (MIM) approach, context autoencoder (CAE), for self-supervised learning. We randomly partition the image into two sets: visible patches and masked patches. The CAE architecture consists of: (i) an encoder that takes visible patches as input and outputs their latent representations, (ii) a latent context regressor that predicts the masked patch representations from the visible patch representations that are not updated in this regressor, (iii) a decoder that takes the estimated masked patch representations as input and makes predictions for the masked patches, and (iv) an alignment module that aligns the masked patch representation estimation with the masked patch representations computed from the encoder. In comparison to previous MIM methods that couple the encoding and decoding roles, e.g., using a single module in BEiT, our approach attempts to separate the encoding role (content understanding) from the decoding role (making predictions for masked patches) using different modules, improving the content understanding capability. In addition, our approach makes predictions from the visible patches to the masked patches in the latent representation space that is expected to take on semantics. In addition, we present the explanations about why contrastive pretraining and supervised pretraining perform similarly and why MIM potentially performs better. We demonstrate the effectiveness of our CAE through superior transfer performance in downstream tasks: semantic segmentation, and object detection and instance segmentation.

<div align="center">
<img src="https://user-images.githubusercontent.com/44519745/195202864-21ab47a2-5dfb-4db1-ac7d-e5f25c29426f.png" width="75%"/>
</div>

## Models and Benchmarks

Here, we report the results provided in the [original repo](https://github.com/lxtGH/CAE), which is pre-trained 300-epoch with ViT-Base on ImageNet-1k. To run the pre-training, please create a new folder `work_dirs/my_pretrains/beit_ckpt` under the root directory and download the
[pretrained weights](https://download.openmmlab.com/mmselfsup/cae/dalle_encoder.pth) for `dalle` encoder to the folder.

| Backbone | Pre-train epoch | Fine-tuning Top-1 |                                                            Pre-train Config                                                            |                                                                        Fine-tuning Config                                                                        |   Download   |
|:--------:|:---------------:|:-----------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------:|
| ViT-Base |       300       |        83.2       | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/cae/imagenet/vit_base_sz224_8xb64_accu4_cos_fp16_ep300.py) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_base_p16_swin_ft_cae_sz224_4xb128_accu2_cos_ep100.py) | model \| log |

## Citation

```bibtex
@article{Chen2022CAE,
  title={Context Autoencoder for Self-Supervised Representation Learning},
  author={Xiaokang Chen and Mingyu Ding and Xiaodi Wang and Ying Xin and Shentong Mo and Yunhao Wang and Shumin Han and Ping Luo and Gang Zeng and Jingdong Wang},
  journal={ArXiv},
  year={2022},
  volume={abs/2202.03026}
}
```
