# SAMix

> [Boosting Discriminative Visual Representation Learning with Scenario-Agnostic Mixup](https://arxiv.org/abs/2111.15454)

## Abstract

Mixup is a popular data-dependent augmentation technique for deep neural networks, which contains two sub-tasks, mixup generation and classification. The community typically confines mixup to supervised learning (SL) and the objective of generation sub-task is fixed to the sampled pairs instead of considering the whole data manifold. To overcome such limitations, we systematically study the objectives of two sub-tasks and propose Scenario-Agostic Mixup for both SL and Self-supervised Learning (SSL) scenarios, named SAMix. Specifically, we hypothesize and verify the core objective of mixup generation as optimizing the local smoothness between two classes subject to global discrimination from other classes. Based on this discovery, η-balanced mixup loss is proposed for complementary training of the two sub-tasks. Meanwhile, the generation sub-task is parameterized as an optimizable module, Mixer, which utilizes an attention mechanism to generate mixed samples without label dependency. Extensive experiments on SL and SSL tasks demonstrate that SAMix consistently outperforms leading methods by a large margin.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/174272657-fb662377-b7c3-4faa-8d9b-ea6f1e08549e.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

We will update configs and models for SAMix soon. Please refer to [Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md) for image classification results.

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
