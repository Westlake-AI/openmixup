# Mixup Classification Benchmark on Places205

> [Places: A 10 million Image Database for Scene Recognition](http://places2.csail.mit.edu/PAMI_places.pdf)

## Abstract

The rise of multi-million-item dataset initiatives has enabled data-hungry machine learning algorithms to reach near-human semantic classification performance at tasks such as visual object and scene recognition. Here we describe the Places Database, a repository of 10 million scene photographs, labeled with scene semantic categories, comprising a large and diverse list of the types of environments encountered in the world. Using the state-of-the-art Convolutional Neural Networks (CNNs), we provide scene classification CNNs (Places-CNNs) as baselines, that significantly outperform the previous approaches. Visualization of the CNNs trained on Places shows that object detectors emerge as an intermediate representation of scene classification. With its high-coverage and high-diversity of exemplars, the Places Database along with the Places-CNNs offer a novel resource to guide future progress on scene recognition problems.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/185649984-7e82d3d5-1ef5-49c5-b08f-06b4d98bb4c5.png" width="100%"/>
</div>

## Results and models

We provide a collection of [weights and logs](https://github.com/Westlake-AI/openmixup/releases/tag/mixup-place205-weights) for mixup classification benchmark on Place205. You can download all results from **Baidu Cloud**: [Places205 (4m94)](https://pan.baidu.com/s/1ciAYxK6SwR13UNScp0W3bQ).

### Getting Started

* You can start training and evaluating with a config file. An example with 4 GPUs on a single node,
  ```shell
  CUDA_VISIBLE_DEVICES=1,2,3,4 PORT=29001 bash tools/dist_train.sh ${CONFIG_FILE} 4
  ```
* Please refer to config files of [Places205](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/place205/) for experiment details. You can modify `max_epochs` and `mix_mode` in `auto_train_mixups.py` to generate configs and bash scripts. Here is an example of using Mixup and CutMix with switching probabilities of $\{0.4, 0.6\}$ based on [base_config](https://github.com/Westlake-AI/openmixup/tree/main/configs/classification/place205/mixups/r50_mixups_CE_none_4xb64.py).
  ```python
  model = dict(
      alpha=[0.8, 1],  # list of alpha
      mix_mode=["mixup", "cutmix"],  # list of chosen mixup modes
      mix_prob=[0.4, 0.6],  # list of applying probs (sum=1), `None` for random applying
      mix_repeat=1,  # times of repeating mixups in each iteration
  )
  ```

### Places-205

**Setup**

* All compared methods adopt ResNet-18/50 architectures and are trained 100 epochs using the PyTorch training recipe. The training and testing image size is 224 with the CenterCrop ratio of 0.85. We search $\alpha$ in $Beta(\alpha, \alpha)$ for all compared methods.
* The **median** of top-1 accuracy in the last 5 training epochs is reported for ResNet-18/50.
* Visualization of mixed samples from [AutoMix](https://arxiv.org/abs/2103.13027) and [SAMix](https://arxiv.org/abs/2111.15454) are provided in zip files.

| Backbones                                                   | ResNet-18 top-1 | ResNet-50 top-1 |
|-------------------------------------------------------------|:---------------:|:---------------:|
| Vanilla                                                     |      59.63      |      63.10      |
| MixUp [[ICLR'2018](https://arxiv.org/abs/1710.09412)]       |      59.33      |      63.01      |
| CutMix [[ICCV'2019](https://arxiv.org/abs/1905.04899)]      |      59.21      |      63.75      |
| ManifoldMix [[ICML'2019](https://arxiv.org/abs/1806.05236)] |      59.46      |      63.23      |
| SaliencyMix [[ICLR'2021](https://arxiv.org/abs/2006.01791)] |      59.50      |      63.33      |
| FMix [[Arixv'2020](https://arxiv.org/abs/2002.12047)]       |      59.51      |      63.63      |
| PuzzleMix [[ICML'2020](https://arxiv.org/abs/2009.06962)]   |      59.62      |      63.91      |
| ResizeMix [[Arixv'2020](https://arxiv.org/abs/2012.11101)]  |      59.66      |      63.88      |
| AutoMix [[ECCV'2022](https://arxiv.org/abs/2103.13027)]     |      59.74      |      64.06      |
| SAMix [[Arxiv'2021](https://arxiv.org/abs/2111.15454)]      |      59.86      |      64.27      |

We summarize mixup benchmarks in [Model Zoo](https://github.com/Westlake-AI/openmixup/tree/main/docs/en/model_zoos/Model_Zoo_sup.md).


## Citation

Please refer to the original paper of [Places205](http://places2.csail.mit.edu/PAMI_places.pdf) for dataset information, and refer to [AutoMix](https://arxiv.org/abs/2103.13027) for experiment details.

```bibtex
@article{zhou2017places,
  title={Places: A 10 million Image Database for Scene Recognition},
  author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2017},
  publisher={IEEE}
}
```
