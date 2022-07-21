# AutoMix

> [AutoMix: Unveiling the Power of Mixup for Stronger Classifiers](https://arxiv.org/abs/2103.13027)

## Abstract

Data mixing augmentation have proved to be effective in improving the generalization ability of deep neural networks. While early methods mix samples by hand-crafted policies (e.g., linear interpolation), recent methods utilize saliency information to match the mixed samples and labels via complex offline optimization. However, there arises a trade-off between precise mixing policies and optimization complexity. To address this challenge, we propose a novel automatic mixup (AutoMix) framework, where the mixup policy is parameterized and serves the ultimate classification goal directly. Specifically, AutoMix reformulates the mixup classification into two sub-tasks (i.e., mixed sample generation and mixup classification) with corresponding sub-networks and solves them in a bi-level optimization framework. For the generation, a learnable lightweight mixup generator, Mix Block, is designed to generate mixed samples by modeling patch-wise relationships under the direct supervision of the corresponding mixed labels. To prevent the degradation and instability of bi-level optimization, we further introduce a momentum pipeline to train AutoMix in an end-to-end manner. Extensive experiments on nine image benchmarks prove the superiority of AutoMix compared with state-of-the-arts in various classification scenarios and downstream tasks.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/174272662-19ce57ad-7b08-4e73-81b1-3bb81fee2fe5.png" width="100%"/>
</div>

## Results and models

### ImageNet-1k

|     Model     |  Mixup  | resolution | Params(M) | Epochs | Top-1 (%) |                               Config                                |                               Download                                |
| :-----------: | :-----: | :--------: | :-------: | :----: | :-------: | :-----------------------------------------------------------------: | :-------------------------------------------------------------------: |
|  ResNet-18    | AutoMix |  224x224   |   11.17   |  100   |   70.50   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix/basic/r18_l2_a2_near_lam_cat_mb_mlr1e_3_bb_mlr0.py) | model / log |
|  ResNet-18    | AutoMix |  224x224   |   11.17   |  300   |   72.05   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix/basic/r18_l2_a2_near_lam_cat_mb_mlr1e_3_bb_mlr0.py) | model / log |
|  ResNet-34    | AutoMix |  224x224   |   21.28   |  100   |   74.52   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix/basic/r34_l2_a2_near_lam_cat_mb_mlr1e_3_bb_mlr0.py) | model / log |
|  ResNet-34    | AutoMix |  224x224   |   21.28   |  300   |   76.10   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix/basic/r34_l2_a2_near_lam_cat_mb_mlr1e_3_bb_mlr0.py) | model / log |
|  ResNet-50    | AutoMix |  224x224   |   23.52   |  100   |   77.91   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix/basic/r50_l2_a2_near_lam_cat_mb_mlr1e_3_bb_mlr0.py) | model / log |
|  ResNet-50    | AutoMix |  224x224   |   23.52   |  300   |   79.25   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix/basic/r50_l2_a2_near_lam_cat_mb_mlr1e_3_bb_mlr0.py) | model / log |
|  ResNet-101   | AutoMix |  224x224   |   42.51   |  100   |   79.87   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix/basic/r101_l2_a2_near_lam_cat_mb_mlr1e_3_bb_mlr0.py) | model / log |
|  ResNet-101   | AutoMix |  224x224   |   42.51   |  300   |   80.98   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix/basic/r101_l2_a2_near_lam_cat_mb_mlr1e_3_bb_mlr0.py) | model / log |
|  ResNeXt-101  | AutoMix |  224x224   |   44.18   |  100   |   80.89   | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/imagenet/automix/basic/rx101_l2_a2_near_lam_cat_mb_mlr1e_3_bb_mlr0.py) | model / log |


We will update configs and models for AutoMix soon. Please refer to [Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md) for image classification results.


## Citation

```bibtex
@misc{liu2021automix,
      title={AutoMix: Unveiling the Power of Mixup for Stronger Classifiers},
      author={Zicheng Liu and Siyuan Li and Di Wu and Zhiyuan Chen and Lirong Wu and Jianzhu Guo and Stan Z. Li},
      year={2021},
      eprint={2103.13027},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
