# Awesome Mixup Methods for Self- and Semi-supervised Learning

 [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![GitHub stars](https://img.shields.io/github/stars/Westlake-AI/openmixup?color=green) ![visitors](https://visitor-badge.glitch.me/badge?page_id=Westlake-AI/openmixup)

**We summarize mixup methods proposed for self- and semi-supervised visual representation learning.**
We are working on a survey of mixup methods. The list is on updating.

## Mixup for Self-supervised Learning

* **MixCo**: Sungnyun Kim, Gihun Lee, Sangmin Bae, Se-Young Yun.
   - MixCo: Mix-up Contrastive Learning for Visual Representation. [[NIPSW'2020](https://arxiv.org/abs/2010.06300)] [[code](https://github.com/Lee-Gihun/MixCo-Mixup-Contrast)]
* **MoCHi**: Yannis Kalantidis, Mert Bulent Sariyildiz, Noe Pion, Philippe Weinzaepfel, Diane Larlus.
   - Hard Negative Mixing for Contrastive Learning. [[NIPS'2020](https://arxiv.org/abs/2010.01028)] [[code](https://europe.naverlabs.com/mochi)]
* **i-Mix**: Kibok Lee, Yian Zhu, Kihyuk Sohn, Chun-Liang Li, Jinwoo Shin, Honglak Lee.
   - i-Mix A Domain-Agnostic Strategy for Contrastive Representation Learning. [[ICLR'2021](https://arxiv.org/abs/2010.08887)] [[code](https://github.com/kibok90/imix)]
* **Un-Mix**: Zhiqiang Shen, Zechun Liu, Zhuang Liu, Marios Savvides, Trevor Darrell, Eric Xing.
   - Un-Mix: Rethinking Image Mixtures for Unsupervised Visual Representation. [[AAAI'2022](https://arxiv.org/abs/2003.05438)] [[code](https://github.com/szq0214/Un-Mix)]
* **BSIM**: Xiangxiang Chu, Xiaohang Zhan, Xiaolin Wei.
   - Beyond Single Instance Multi-view Unsupervised Representation Learning. [[Arxiv'2020](https://arxiv.org/abs/2011.13356)]
* **FT**: Rui Zhu, Bingchen Zhao, Jingen Liu, Zhenglong Sun, Chang Wen Chen.
   - Improving Contrastive Learning by Visualizing Feature Transformation. [[ICCV'2021](https://arxiv.org/abs/2108.02982)] [[code](https://github.com/DTennant/CL-Visualizing-Feature-Transformation)]
* **m-Mix**: Shaofeng Zhang, Meng Liu, Junchi Yan, Hengrui Zhang, Lingxiao Huang, Pinyan Lu, Xiaokang Yang.
   - m-mix: Generating hard negatives via multiple samples mixing for contrastive learning. [[Arxiv'2021](https://openreview.net/forum?id=lsljy2bG3n)]
* **PCEA**: Jingwei Liu, Yi Gu, Shentong Mo, Zhun Sun, Shumin Han, Jiafeng Guo, Xueqi Cheng.
   - Piecing and Chipping: An effective solution for the information-erasing view generation in Self-supervised Learning. [[OpenReview'2021](https://openreview.net/forum?id=DnG8f7gweH4)]
* **CoMix**: Aadarsh Sahoo, Rutav Shah, Rameswar Panda, Kate Saenko, Abir Das.
   - Contrast and Mix: Temporal Contrastive Video Domain Adaptation with Background Mixing. [[NIPS'2021](https://proceedings.neurips.cc/paper/2021/file/c47e93742387750baba2e238558fa12d-Paper.pdf)] [[code](https://cvir.github.io/projects/comix)]
* **SAMix**: Siyuan Li, Zicheng Liu, Di Wu, Zihan Liu, Stan Z. Li.
   - Boosting Discriminative Visual Representation Learning with Scenario-Agnostic Mixup. [[Arxiv'2021](https://arxiv.org/abs/2111.15454)] [[code](https://github.com/Westlake-AI/openmixup)]
* **MixSiam**: Xiaoyang Guo, Tianhao Zhao, Yutian Lin, Bo Du.
   - MixSiam: A Mixture-based Approach to Self-supervised Representation Learning. [[OpenReview'2021](https://arxiv.org/abs/2111.02679)]
* **MixSSL**: Yichen Zhang, Yifang Yin, Ying Zhang, Roger Zimmermann.
   - Mix-up Self-Supervised Learning for Contrast-agnostic Applications. [[ICME'2021](https://arxiv.org/abs/2204.00901)]
* **CLIM**: Hao Li, Xiaopeng Zhang, Hongkai Xiong.
   - Center-wise Local Image Mixture For Contrastive Representation Learning. [[BMVC'2021](https://arxiv.org/abs/2011.02697)]
* **Mixup** Xin Zhang, Minho Jin, Roger Cheng, Ruirui Li, Eunjung Han, Andreas Stolcke.
   - Contrastive-mixup Learning for Improved Speaker Verification. [[ICASSP'2022](https://arxiv.org/abs/2202.10672)]
* **Metrix**: Shashanka Venkataramanan, Bill Psomas, Ewa Kijak, Laurent Amsaleg, Konstantinos Karantzalos, Yannis Avrithis.
   - It Takes Two to Tango: Mixup for Deep Metric Learning. [[ICLR'2022](https://arxiv.org/abs/2106.04990)] [[code](https://github.com/billpsomas/Metrix_ICLR22)]
* **ProGCL** Jun Xia, Lirong Wu, Ge Wang, Jintao Chen, Stan Z.Li.
   - ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning. [[ICML'2022](https://arxiv.org/abs/2110.02027)] [[code](https://github.com/junxia97/ProGCL)]
* **M-Mix** Shaofeng Zhang, Meng Liu, Junchi Yan, Hengrui Zhang, Lingxiao Huang, Pinyan Lu, Xiaokang Yang.
   - M-Mix: Generating Hard Negatives via Multi-sample Mixing for Contrastive Learning. [[KDD'2022](https://sherrylone.github.io/assets/KDD22_M-Mix.pdf)] [[code](https://github.com/Sherrylone/m-mix)]
* **SDMP**: Sucheng Ren, Huiyu Wang, Zhengqi Gao, Shengfeng He, Alan Yuille, Yuyin Zhou, Cihang Xie.
   - A Simple Data Mixing Prior for Improving Self-Supervised Learning. [[CVPR'2022](https://arxiv.org/abs/2206.07692)] [[code](https://github.com/oliverrensu/sdmp)]
* **ScaleMix**: Xiao Wang, Haoqi Fan, Yuandong Tian, Daisuke Kihara, Xinlei Chen.
   - On the Importance of Asymmetry for Siamese Representation Learning. [[CVPR'2022](https://arxiv.org/abs/2204.00613)] [[code](https://github.com/facebookresearch/asym-siam)]
* **VLMixer**: Teng Wang, Wenhao Jiang, Zhichao Lu, Feng Zheng, Ran Cheng, Chengguo Yin, Ping Luo.
   - VLMixer: Unpaired Vision-Language Pre-training via Cross-Modal CutMix. [[ICML'2022](https://arxiv.org/abs/2206.08919)]
* **CropMix**: Junlin Han, Lars Petersson, Hongdong Li, Ian Reid.
   - CropMix: Sampling a Rich Input Distribution via Multi-Scale Cropping. [[ArXiv'2022](https://arxiv.org/abs/2205.15955)] [[code](https://github.com/JunlinHan/CropMix)]


## Mixup for Semi-supervised Learning

* **MixMatch**: David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, Colin Raffel.
   - MixMatch: A Holistic Approach to Semi-Supervised Learning. [[NIPS'2019](https://arxiv.org/abs/1905.02249)] [[code](https://github.com/google-research/mixmatch)]
* **Pani VAT**: Ke Sun, Bing Yu, Zhouchen Lin, Zhanxing Zhu.
   - Patch-level Neighborhood Interpolation: A General and Effective Graph-based Regularization Strategy. [[ArXiv'2019](https://arxiv.org/abs/1911.09307)]
* **ReMixMatch**: David Berthelot, dberth@google.com, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, Colin Raffel.
   - ReMixMatch: Semi-Supervised Learning with Distribution Matching and Augmentation Anchoring. [[ICLR'2020](https://openreview.net/forum?id=HklkeR4KPB)] [[code](https://github.com/google-research/remixmatch)]
* **Core-Tuning**: Yifan Zhang, Bryan Hooi, Dapeng Hu, Jian Liang, Jiashi Feng.
   - Unleashing the Power of Contrastive Self-Supervised Visual Models via Contrast-Regularized Fine-Tuning. [[NIPS'2021](https://arxiv.org/abs/2102.06605)] [[code](https://github.com/vanint/core-tuning)]
* **DFixMatch**: Zicheng Liu, Siyuan Li, Ge Wang, Cheng Tan, Lirong Wu, Stan Z. Li.
   - Decoupled Mixup for Data-efficient Learning. [[Arxiv'2022](https://arxiv.org/abs/2203.10761)] [[code](https://github.com/Westlake-AI/openmixup)]


## Contribution

Feel free to send [pull requests](https://github.com/Westlake-AI/openmixup/pulls) to add more links with the following Markdown format. Current contributors include: Siyuan Li ([@Lupin1998](https://github.com/Lupin1998)) and Zicheng Liu ([@pone7](https://github.com/pone7)).

```markdown
* **Abbreviation**: Author List.
  - Paper Name. [[Conference'Year](link)] [[code](link)]
```
