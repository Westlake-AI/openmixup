# Model Zoo of Self-supervised Learning

**Current results of self-supervised learning benchmarks are based on [MMSelfSup](https://github.com/open-mmlab/mmselfsup) and [solo-learn](https://github.com/vturrisi/solo-learn). We will rerun the experiments and update more reliable results soon!**

<details open>
<summary>Currently supported self-supervised learning methods</summary>

- [x] [Relative Location [ICCV'2015]](https://arxiv.org/abs/1505.05192)
- [x] [Rotation Prediction [ICLR'2018]](https://arxiv.org/abs/1803.07728)
- [x] [DeepCluster [ECCV'2018]](https://arxiv.org/abs/1807.05520)
- [x] [NPID [CVPR'2018]](https://arxiv.org/abs/1805.01978)
- [x] [ODC [CVPR'2020]](https://arxiv.org/abs/2006.10645)
- [x] [MoCov1 [CVPR'2020]](https://arxiv.org/abs/1911.05722)
- [x] [SimCLR [ICML'2020]](https://arxiv.org/abs/2002.05709)
- [x] [MoCov2 [ArXiv'2020]](https://arxiv.org/abs/2003.04297)
- [x] [BYOL [NIPS'2020]](https://arxiv.org/abs/2006.07733)
- [x] [SwAV [NIPS'2020]](https://arxiv.org/abs/2006.09882)
- [x] [DenseCL [CVPR'2021]](https://arxiv.org/abs/2011.09157)
- [x] [SimSiam [CVPR'2021]](https://arxiv.org/abs/2011.10566)
- [x] [Barlow Twins [ICML'2021]](https://arxiv.org/abs/2103.03230)
- [x] [MoCo v3 [ICCV'2021]](https://arxiv.org/abs/2104.02057)
- [x] [MAE [CVPR'2022]](https://arxiv.org/abs/2111.06377)
- [x] [SimMIM [CVPR'2022]](https://arxiv.org/abs/2111.09886)
- [x] [CAE [ArXiv'2022]](https://arxiv.org/abs/2202.03026)
- [x] [A2MIM [ArXiv'2022]](https://arxiv.org/abs/2205.13943)

</details>

## ImageNet-1K pre-trained models

The training details are provided in the config files. You can click the method's name to obtain more information.

| Method | Config | Download |
|-----|-----|:---:|
| [Relative Location](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/relative_loc/README.md) | [r50_8xb64_step_ep70](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/relative_loc/imagenet/r50_8xb64_step_ep70.py) | [model](https://download.openmmlab.com/mmselfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k_20220225-84784688.pth) |
| [Rotation Prediction](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/rotation_pred/README.md) | [r50_8xb64_step_ep70](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/rotation_pred/imagenet/r50_8xb64_step_ep70.py) | [model](https://download.openmmlab.com/mmselfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k_20220225-5b9f06a0.pth) |
| [DeepCluster](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/deepcluster/README.md) | [r50_sobel_8xb64_step_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/deepcluster/imagenet/r50_sobel_8xb64_step_ep200.py) | [model](https://download.openmmlab.com/mmselfsup/deepcluster/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k-bb8681e2.pth) |
| [NPID](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/npid/README.md) | [r50_4xb64_step_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/npid/imagenet/r50_4xb64_step_ep200.py) | [model](https://download.openmmlab.com/mmselfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k_20220225-5fbbda2a.pth) |
| [ODC](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/odc/README.md) | [r50_8xb64_step_ep440](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/odc/imagenet/r50_8xb64_step_ep440.py) | [model](https://download.openmmlab.com/mmselfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k_20220225-a755d9c0.pth) |
| [SimCLR](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simclr/README.md) | [r50_8xb64_cos_lr0_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simclr/imagenet/r50_8xb64_cos_lr0_6_fp16_ep200.py) | [model](https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_20220428-46ef6bb9.pth) |
|   | [r50_16xb256_cos_lr4_8_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simclr/imagenet/r50_16xb256_cos_lr4_8_fp16_ep200.py) | [model](https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_16xb256-coslr-200e_in1k_20220428-8c24b063.pth) |
| [MoCoV2](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov2/README.md) | [r50_4xb64_cos](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov2/imagenet/r50_4xb64_cos.py) | [model](https://download.openmmlab.com/mmselfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k_20220225-89e03af4.pth) |
| [BYOL](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/README.md) | [r50_8xb64_accu8_cos_lr4_8_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/imagenet/r50_8xb64_accu8_cos_lr4_8_fp16_ep200.py) | [model](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_16xb256-coslr-200e_in1k_20220527-b6f8eedd.pth) |
|   | [r50_8xb64_accu8_cos_lr4_8_fp16_ep300](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/byol/imagenet/r50_8xb64_accu8_cos_lr4_8_fp16_ep300.py) | [model](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k_20220225-a0daa54a.pth) |
| [SwAV](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/swav/README.md) | [r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/swav/imagenet/r50_8xb64_accu8_cos_lr9_6-mcrop-224_2-96_6_fp16_ep200.py) | [model](https://download.openmmlab.com/mmselfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220225-0497dd5d.pth) |
| [DenseCL](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/densecl/README.md) | [r50_4xb64_cos](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/densecl/imagenet/r50_4xb64_cos.py) | [model](https://download.openmmlab.com/mmselfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k_20220225-8c7808fe.pth) |
| [SimSiam](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/README.md) | [r50_4xb64_cos_lr0_05_ep100](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep100.py) | [model](https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k_20220225-68a88ad8.pth) |
|   | [r50_4xb64_cos_lr0_05_ep200](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simsiam/imagenet/r50_4xb64_cos_lr0_05_ep200.py) | [model](https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k_20220225-2f488143.pth) |
| [BarlowTwins](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/barlowtwins/README.md) | [r50_8xb64_accu4_cos_lr1_6_ep300](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/barlowtwins/imagenet/r50_8xb64_accu4_cos_lr1_6_ep300.py) | [model](https://download.openmmlab.com/mmselfsup/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k_20220419-5ae15f89.pth) |
| [MoCoV3](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov3/README.md) | [vit_small_8xb64_accu8_cos_fp16_ep300](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mocov3/imagenet/vit_small_8xb64_accu8_cos_fp16_ep300.py) | [model](https://download.openmmlab.com/mmselfsup/moco/mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224_20220225-e31238dd.pth) |
| [MAE](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mae/README.md) | [vit_base_dec8_dim512_8xb128_accu4_cos_fp16_ep400](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/mae/imagenet/vit_base_dec8_dim512_8xb128_accu4_cos_fp16_ep400.py) | [model](https://download.openmmlab.com/mmselfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k-224_20220223-85be947b.pth) |
| [SimMIM](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/README.md) | [swin_base_sz192_8xb128_accu2_cos_ep100](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/swin_base_sz192_8xb128_accu2_cos_ep100.py) | [model](https://download.openmmlab.com/mmselfsup/simmim/simmim_swin-base_16xb128-coslr-100e_in1k-192_20220316-1d090125.pth) |
|   | [vit_base_rgb_m_sz224_8xb128_accu2_step_fp16_ep800](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/vit_base_rgb_m_sz224_8xb128_accu2_step_fp16_ep800.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/full_simmim_vit_base_rgb_m_sz224_8xb128_accu2_step_fp16_ep800.pth) |
| [MaskFeat](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/maskfeat/README.md) | [vit_base_hog_108_sz224_8xb128_accu2_cos_fp16_ep300](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/maskfeat/imagenet/vit_base_hog_108_sz224_8xb128_accu2_cos_fp16_ep300.py) | [model](https://download.openmmlab.com/mmselfsup/maskfeat/maskfeat_vit-base-p16_8xb256-coslr-300e_in1k_20220913-591d4c4b.pth) |
| [CAE](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/cae/RAEDME.md) | [vit_base_sz224_8xb64_accu4_cos_fp16_ep300](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/cae/imagenet/vit_base_sz224_8xb64_accu4_cos_fp16_ep300.py) | [model](https://download.openmmlab.com/mmselfsup/cae/cae_vit-base-p16_16xb256-coslr-300e_in1k-224_20220427-4c786349.pth) |
| [A2MIM](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/README.md) | [r50_l3_sz224_init_8xb256_cos_ep100](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/r50_l3_sz224_init_8xb256_cos_ep100.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/full_a2mim_r50_l3_sz224_init_8xb256_cos_ep100.pth) |
|   | [r50_l3_sz224_init_8xb256_cos_ep300](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/r50_l3_sz224_init_8xb256_cos_ep300.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/full_a2mim_r50_l3_sz224_init_8xb256_cos_ep300.pth) |
|   | [vit_base_l0_sz224_8xb128_accu2_step_ep800](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/vit_base_l0_sz224_8xb128_accu2_step_ep800.py) | [model](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/full_a2mim_vit_base_l0_res_fft01_sz224_4xb128_accu4_step_fp16_ep800.pth) |


### ImageNet-1K Linear Evaluation

**Note**
* If not specifically indicated, the testing GPUs are NVIDIA Tesla V100 on [MMSelfSup](https://github.com/open-mmlab/mmselfsup) and [OpenMixup](https://github.com/Westlake-AI/openmixup). The pre-training and fine-tuning testing image size are $224\times 224$.
* The table records the implementors who implemented the methods (either by themselves or refactoring from other repos), and the experimenters who performed experiments and reproduced the results. The experimenters should be responsible for the evaluation results on all the benchmarks, and the implementors should be responsible for the implementation as well as the results; If the experimenter is not indicated, an implementator is the experimenter by default.
* We use config [r50_multihead](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_mhead_sz224_4xb64_step_ep90.py) for ImageNet multi-heads and [r50_linear](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_mhead_sz224_4xb64_step_ep90.py) for the global average pooled feature evaluation.

| Methods | Remarks | Batch size | Epochs | Procotol | Linear |
|---|---|:---:|:---:|:---:|:---:|
| PyTorch | torchvision | 256 | 90 | MoCo | 76.17 |
| Random | kaiming | - | - | MoCo | 4.35 |
| [Relative-Loc](https://arxiv.org/abs/1505.05192) | ResNet-50 | 512 | 70 | MoCo | 38.83 |
| [Rotation](https://arxiv.org/abs/1803.07728) | ResNet-50 | 128 | 70 | MoCo | 47.01 |
| [DeepCluster](https://arxiv.org/abs/1807.05520) | ResNet-50 | 512 | 200 | MoCo | 46.92 |
| [NPID](https://arxiv.org/abs/1805.01978) | ResNet-50 | 256 | 200 | MoCo | 56.60 |
| [ODC](https://arxiv.org/abs/2006.10645) | ResNet-50 | 512 | 440 | MoCo | 53.42 |
| [SimCLR](https://arxiv.org/abs/2002.05709) | ResNet-50 | 256 | 200 | SimSiam | 62.56 |
|  | ResNet-50 | 4096 | 200 | SimSiam | 66.66 |
| [MoCov1](https://arxiv.org/abs/1911.05722) | ResNet-50 | 256 | 200 | MoCo | 61.02 |
| [MoCoV2](https://arxiv.org/abs/2003.04297) | ResNet-50 | 256 | 200 | MoCo | 67.69 |
| [BYOL](https://arxiv.org/abs/2006.07733) | ResNet-50 | 4096 | 200 | SimSiam | 71.88 |
|  | ResNet-50 | 4096 | 300 | SimSiam | 72.93 |
| [SwAV](https://arxiv.org/abs/2006.09882) | ResNet-50 | 512 | 200 | SimSiam | 70.47 |
| [DenseCL](https://arxiv.org/abs/2011.09157) | ResNet-50 | 256 | 200 | MoCo | 63.62 |
| [SimSiam](https://arxiv.org/abs/2011.10566) | ResNet-50 | 512 | 100 | SimSiam | 68.28 |
|  | ResNet-50 | 512 | 200 | SimSiam | 69.84 |
| [BarlowTwins](https://arxiv.org/abs/2103.03230) | ResNet-50 | 2048 | 300 | BarlowTwins | 71.66 |
| [MoCoV3](https://arxiv.org/abs/2104.02057) | ViT-Small | 4096 | 400 | MoCoV3 | 73.19 |


### ImageNet-1K End-to-end Fine-tuning Evaluation

**Note**
* All compared methods adopt ResNet-50 or ViT-B architectures and are pre-trained on ImageNet-1K. The pre-training and fine-tuning testing image size are $224\times 224$, except for SimMIM with Swin-Base using $192\times 192$. The fine-tuning protocols include: [RSB A3](https://arxiv.org/abs/2110.00476) and [RSB A2](https://arxiv.org/abs/2110.00476) for ResNet-50, [BEiT](https://arxiv.org/abs/2106.08254) for ViT-B.
* You can find pre-training codes of compared methods in [OpenMixup](https://github.com/Westlake-AI/openmixup), [VISSL](https://github.com/facebookresearch/vissl), [solo-learn](https://github.com/vturrisi/solo-learn), and the official repositories. You can download fine-tuned models from [a2mim-in1k-weights](https://github.com/Westlake-AI/openmixup/releases/tag/a2mim-in1k-weights) or [Baidu Cloud (3q5i)](https://pan.baidu.com/s/1aj3Lbj_wvyV_1BRzFhPcwQ?pwd=3q5i).

| Methods | Backbone | Source | Batch size | PT epoch | FT protocol | FT top-1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| PyTorch | ResNet-50 | PyTorch | 256 | 90 | RSB A3 | 78.8 |
| [Inpainting](https://arxiv.org/abs/1604.07379) | ResNet-50 | OpenMixup | 512 | 70 | RSB A3 | 78.4 |
| [Relative-Loc](https://arxiv.org/abs/1505.05192) | ResNet-50 | OpenMixup | 512 | 70 | RSB A3 | 77.8 |
| [Rotation](https://arxiv.org/abs/1803.07728) | ResNet-50 | OpenMixup | 128 | 70 | RSB A3 | 77.7 |
| [SimCLR](https://arxiv.org/abs/2002.05709) | ResNet-50 | VISSL | 4096 | 100 | RSB A3 | 78.5 |
| [MoCoV2](https://arxiv.org/abs/2003.04297) | ResNet-50 | OpenMixup | 256 | 100 | RSB A3 | 78.5 |
| [BYOL](https://arxiv.org/abs/2006.07733) | ResNet-50 | OpenMixup | 4096 | 100 | RSB A3 | 78.7 |
|  | ResNet-50 | Official | 4096 | 300 | RSB A3 | 78.9 |
|  | ResNet-50 | Official | 4096 | 300 | RSB A2 | 80.1 |
| [SwAV](https://arxiv.org/abs/2006.09882) | ResNet-50 | VISSL | 4096 | 100 | RSB A3 | 78.9 |
|  | ResNet-50 | Official | 4096 | 400 | RSB A3 | 79.0 |
|  | ResNet-50 | Official | 4096 | 400 | RSB A2 | 80.2 |
| [BarlowTwins](https://arxiv.org/abs/2103.03230) | ResNet-50 | solo learn | 2048 | 100 | RSB A3 | 78.5 |
|  | ResNet-50 | Official | 2048 | 300 | RSB A3 | 78.8 |
| [MoCoV3](https://arxiv.org/abs/2104.02057) | ResNet-50 | Official | 4096 | 100 | RSB A3 | 78.7 |
|  | ResNet-50 | Official | 4096 | 300 | RSB A3 | 79.0 |
|  | ResNet-50 | Official | 4096 | 300 | RSB A2 | 80.1 |
| [A2MIM](https://arxiv.org/abs/2205.13943) | ResNet-50 | OpenMixup | 2048 | 100 | RSB A3 | 78.8 |
|  | ResNet-50 | OpenMixup | 2048 | 300 | RSB A3 | 78.9 |
|  | ResNet-50 | OpenMixup | 2048 | 300 | RSB A2 | 80.4 |
| [MAE](https://arxiv.org/abs/2111.06377) | ViT-Base | OpenMixup | 4096 | 400 | BEiT (MAE) | 83.1 |
| [SimMIM](https://arxiv.org/abs/2111.09886) | Swin-Base | OpenMixup | 2048 | 100 | BEiT (SimMIM) | 82.9 |
|  | ViT-Base | OpenMixup | 2048 | 800 | BEiT (SimMIM) | 83.9 |
| [CAE](https://arxiv.org/abs/2202.03026) | ViT-Base | OpenMixup | 2048 | 300 | BEiT (CAE) | 83.2 |
| [MaskFeat](https://arxiv.org/abs/2112.09133) | ViT-Base | OpenMixup | 2048 | 300 | BEiT (MaskFeat) | 83.5 |
| [A2MIM](https://arxiv.org/abs/2205.13943) | ViT-Base | OpenMixup | 2048 | 800 | BEiT (SimMIM) | 84.3 |


## Downstream Task Benchmarks

### Places205 Linear Classification

**Note**
* In this benchmark, we use the config files of [r50_mhead](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/place205/r50_mhead_sz224_4xb64_step_ep28.py) and [r50_mhead_sobel](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/place205/r50_sobel_mhead_sz224_4xb64_step_ep28.py). For DeepCluster, use the corresponding one with `_sobel`.
* Places205 evaluates features in around 9k dimensions from different layers. Top-1 result of the last epoch is reported.


### ImageNet Semi-Supervised Classification

**Note**
* In this benchmark, the necks or heads are removed and only the backbone CNN is evaluated by appending a linear classification head. All parameters are fine-tuned. We use config files under [imagenet_per_1](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/imagenet_per_1) for 1% data and [imagenet_per_10](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/imagenet_per_10) for 10% data.
* When training with 1% ImageNet, we find hyper-parameters especially the learning rate greatly influence the performance. Hence, we prepare a list of settings with the base learning rate from \{0.001, 0.01, 0.1\} and the learning rate multiplier for the head from \{1, 10, 100\}. We choose the best performing setting for each method. Please use `--deterministic` in this benchmark.


### PASCAL VOC07+12 Object Detection

**Note**
* This benchmark follows the evluation protocols set up by MoCo. [model_zoo](https://github.com/open-mmlab/mmselfsup/blob/master/docs/en/model_zoo.md) in [MMSelfSup](https://github.com/open-mmlab/mmselfsup) for results.
* Config: `benchmarks/detection/configs/pascal_voc_R_50_C4_24k_moco.yaml`.
* Please follow [here](docs/en/get_started.md#voc0712--coco17-object-detection) to run the evaluation.


### COCO2017 Object Detection

**Note**
* This benchmark follows the evluation protocols set up by MoCo. Refer to [model_zoo](https://github.com/open-mmlab/mmselfsup/blob/master/docs/en/model_zoo.md) in [MMSelfSup](https://github.com/open-mmlab/mmselfsup) for results.
* Config: `benchmarks/detection/configs/coco_R_50_C4_2x_moco.yaml`.
* Please follow [here](docs/en/get_started.md#voc0712--coco17-object-detection) to run the evaluation.
