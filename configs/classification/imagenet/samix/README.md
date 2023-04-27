# SAMix

> [Boosting Discriminative Visual Representation Learning with Scenario-Agnostic Mixup](https://arxiv.org/abs/2111.15454)

## Abstract

Mixup is a popular data-dependent augmentation technique for deep neural networks, which contains two sub-tasks, mixup generation and classification. The community typically confines mixup to supervised learning (SL) and the objective of generation sub-task is fixed to the sampled pairs instead of considering the whole data manifold. To overcome such limitations, we systematically study the objectives of two sub-tasks and propose Scenario-Agostic Mixup for both SL and Self-supervised Learning (SSL) scenarios, named SAMix. Specifically, we hypothesize and verify the core objective of mixup generation as optimizing the local smoothness between two classes subject to global discrimination from other classes. Based on this discovery, η-balanced mixup loss is proposed for complementary training of the two sub-tasks. Meanwhile, the generation sub-task is parameterized as an optimizable module, Mixer, which utilizes an attention mechanism to generate mixed samples without label dependency. Extensive experiments on SL and SSL tasks demonstrate that SAMix consistently outperforms leading methods by a large margin.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/174272657-fb662377-b7c3-4faa-8d9b-ea6f1e08549e.png" width="70%"/>
</div>

## Results and models

### ImageNet-1k

|     Model     |  Mixup  | resolution | Params(M) | Epochs | Top-1 (%) |                               Config                                |                               Download                                |
| :-----------: | :-----: | :--------: | :-------: | :----: | :-------: | :-----------------------------------------------------------------: | :-------------------------------------------------------------------: |
|  ResNet-18    |  SAMix  |  224x224   |   11.17   |  100   |   70.50   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix/basic/r18_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0_4xb64.py) | model / log |
|  ResNet-18    |  SAMix  |  224x224   |   11.17   |  300   |   72.05   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix/basic/r18_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0_4xb64.py) | model / log |
|  ResNet-34    |  SAMix  |  224x224   |   21.28   |  100   |   74.52   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix/basic/r34_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0_4xb64.py) | model / log |
|  ResNet-34    |  SAMix  |  224x224   |   21.28   |  300   |   76.10   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix/basic/r34_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0_4xb64.py) | model / log |
|  ResNet-50    |  SAMix  |  224x224   |   23.52   |  100   |   77.91   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix/basic/r50_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0_4xb64.py) | model / log |
|  ResNet-50    |  SAMix  |  224x224   |   23.52   |  300   |   79.25   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix/basic/r50_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0_4xb64.py) | model / log |
|  ResNet-101   |  SAMix  |  224x224   |   42.51   |  100   |   79.87   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix/basic/r101_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0_4xb64.py) | model / log |
|  ResNet-101   |  SAMix  |  224x224   |   42.51   |  300   |   80.98   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix/basic/r101_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0_4xb64.py) | model / log |
|  ResNeXt-101  |  SAMix  |  224x224   |   44.18   |  100   |   80.89   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/samix/basic/rx101_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0_4xb64.py) | model / log |

We will update configs and models (ResNets, ViTs, Swin-T, and ConvNeXt-T) for SAMix soon (please contact us if you want the models right now). Please refer to [Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md) for image classification results.

## Citation

```bibtex
@misc{li2021samix,
      title={Boosting Discriminative Visual Representation Learning with Scenario-Agnostic Mixup},
      author={Siyuan Li and Zicheng Liu and Di Wu and Zihan Liu and Stan Z. Li},
      year={2021},
      eprint={2111.15454},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
