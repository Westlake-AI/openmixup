# Awesome Mixup Methods for Self- and Semi-supervised Learning

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![GitHub stars](https://img.shields.io/github/stars/Westlake-AI/openmixup?color=blue) ![GitHub forks](https://img.shields.io/github/forks/Westlake-AI/openmixup?color=yellow&label=Fork)

**We summarize mixup methods proposed for self- and semi-supervised visual representation learning.**
We are working on a survey of mixup methods. The list is on updating.

* To find related papers and their relationships, check out [Connected Papers](https://www.connectedpapers.com/), which visualizes the academic field in a graph representation.
* To export BibTeX citations of papers, check out [ArXiv](https://arxiv.org/) or [Semantic Scholar](https://www.semanticscholar.org/) of the paper for professional reference formats.

## Table of Contents

  - [Mixup for Self-supervised Learning](#mixup-for-self-supervised-learning)
  - [Mixup for Semi-supervised Learning](#mixup-for-semi-supervised-learning)
  - [Contribution](#contribution)


## Mixup for Self-supervised Learning

* **MixCo**: Sungnyun Kim, Gihun Lee, Sangmin Bae, Se-Young Yun.
   - MixCo: Mix-up Contrastive Learning for Visual Representation. [[NIPSW'2020](https://arxiv.org/abs/2010.06300)] [[code](https://github.com/Lee-Gihun/MixCo-Mixup-Contrast)]
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204580767-c0730ac6-802f-40bc-92a8-4b7abe0acb99.png" /></p>
* **MoCHi**: Yannis Kalantidis, Mert Bulent Sariyildiz, Noe Pion, Philippe Weinzaepfel, Diane Larlus.
   - Hard Negative Mixing for Contrastive Learning. [[NIPS'2020](https://arxiv.org/abs/2010.01028)] [[code](https://europe.naverlabs.com/mochi)]
   <p align="center"><img width="40%" src="https://user-images.githubusercontent.com/44519745/204580935-80ebaaaa-2761-4da4-9b58-7755a0dc15c6.png" /></p>
* **i-Mix**: Kibok Lee, Yian Zhu, Kihyuk Sohn, Chun-Liang Li, Jinwoo Shin, Honglak Lee.
   - i-Mix A Domain-Agnostic Strategy for Contrastive Representation Learning. [[ICLR'2021](https://arxiv.org/abs/2010.08887)] [[code](https://github.com/kibok90/imix)]
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204581084-5460bd37-4adb-4f01-b7af-fc88ceb2683e.png" /></p>
* **Un-Mix**: Zhiqiang Shen, Zechun Liu, Zhuang Liu, Marios Savvides, Trevor Darrell, Eric Xing.
   - Un-Mix: Rethinking Image Mixtures for Unsupervised Visual Representation. [[AAAI'2022](https://arxiv.org/abs/2003.05438)] [[code](https://github.com/szq0214/Un-Mix)]
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204581434-dfbc11f8-e300-4dd7-bc84-adbe5a53dbf4.png" /></p>
* **BSIM**: Xiangxiang Chu, Xiaohang Zhan, Xiaolin Wei.
   - Beyond Single Instance Multi-view Unsupervised Representation Learning. [[BMVC'2022](https://arxiv.org/abs/2011.13356)]
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204581834-3ead412b-359c-40ba-86ea-3ab54ead2c96.png" /></p>
* **FT**: Rui Zhu, Bingchen Zhao, Jingen Liu, Zhenglong Sun, Chang Wen Chen.
   - Improving Contrastive Learning by Visualizing Feature Transformation. [[ICCV'2021](https://arxiv.org/abs/2108.02982)] [[code](https://github.com/DTennant/CL-Visualizing-Feature-Transformation)]
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204582514-426efca3-4e8b-48b8-b69a-ecebb94b7fa8.png" /></p>
* **PCEA**: Jingwei Liu, Yi Gu, Shentong Mo, Zhun Sun, Shumin Han, Jiafeng Guo, Xueqi Cheng.
   - Piecing and Chipping: An effective solution for the information-erasing view generation in Self-supervised Learning. [[OpenReview'2021](https://openreview.net/forum?id=DnG8f7gweH4)]
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204582781-b54c4472-67eb-4e78-9362-44b990bbafa3.png" /></p>
* **CoMix**: Aadarsh Sahoo, Rutav Shah, Rameswar Panda, Kate Saenko, Abir Das.
   - Contrast and Mix: Temporal Contrastive Video Domain Adaptation with Background Mixing. [[NIPS'2021](https://proceedings.neurips.cc/paper/2021/file/c47e93742387750baba2e238558fa12d-Paper.pdf)] [[code](https://cvir.github.io/projects/comix)]
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204583146-fc363695-6889-46cd-93f1-236bec9d5fb5.png" /></p>
* **SAMix**: Siyuan Li, Zicheng Liu, Di Wu, Zihan Liu, Stan Z. Li.
   - Boosting Discriminative Visual Representation Learning with Scenario-Agnostic Mixup. [[Arxiv'2021](https://arxiv.org/abs/2111.15454)] [[code](https://github.com/Westlake-AI/openmixup)]
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/174272657-fb662377-b7c3-4faa-8d9b-ea6f1e08549e.png" /></p>
* **MixSiam**: Xiaoyang Guo, Tianhao Zhao, Yutian Lin, Bo Du.
   - MixSiam: A Mixture-based Approach to Self-supervised Representation Learning. [[OpenReview'2021](https://arxiv.org/abs/2111.02679)]
   <p align="center"><img width="60%" src="https://user-images.githubusercontent.com/44519745/204583448-8c1890fd-ce95-488a-9570-f7393a4d140a.png" /></p>
* **MixSSL**: Yichen Zhang, Yifang Yin, Ying Zhang, Roger Zimmermann.
   - Mix-up Self-Supervised Learning for Contrast-agnostic Applications. [[ICME'2021](https://arxiv.org/abs/2204.00901)]
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204583625-58632669-05f0-445b-bd41-60aa37b515d4.png" /></p>
* **CLIM**: Hao Li, Xiaopeng Zhang, Hongkai Xiong.
   - Center-wise Local Image Mixture For Contrastive Representation Learning. [[BMVC'2021](https://arxiv.org/abs/2011.02697)]
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204583930-15ab3916-9d8b-4adf-9a79-e40eabbbc255.png" /></p>
* **Mixup** Xin Zhang, Minho Jin, Roger Cheng, Ruirui Li, Eunjung Han, Andreas Stolcke.
   - Contrastive-mixup Learning for Improved Speaker Verification. [[ICASSP'2022](https://arxiv.org/abs/2202.10672)]
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204584821-5ef0cdd9-37bf-437a-a139-d21068452be1.png" /></p>
* **Metrix**: Shashanka Venkataramanan, Bill Psomas, Ewa Kijak, Laurent Amsaleg, Konstantinos Karantzalos, Yannis Avrithis.
   - It Takes Two to Tango: Mixup for Deep Metric Learning. [[ICLR'2022](https://arxiv.org/abs/2106.04990)] [[code](https://github.com/billpsomas/Metrix_ICLR22)]
   <p align="center"><img width="45%" src="https://user-images.githubusercontent.com/44519745/204584574-9f090728-29c6-4c06-8e50-8344c789983f.png" /></p>
* **ProGCL** Jun Xia, Lirong Wu, Ge Wang, Jintao Chen, Stan Z.Li.
   - ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning. [[ICML'2022](https://arxiv.org/abs/2110.02027)] [[code](https://github.com/junxia97/ProGCL)]
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/204584450-e31a6c9f-0f1c-4342-b907-1b80cae547ab.png" /></p>
* **M-Mix** Shaofeng Zhang, Meng Liu, Junchi Yan, Hengrui Zhang, Lingxiao Huang, Pinyan Lu, Xiaokang Yang.
   - M-Mix: Generating Hard Negatives via Multi-sample Mixing for Contrastive Learning. [[KDD'2022](https://sherrylone.github.io/assets/KDD22_M-Mix.pdf)] [[code](https://github.com/Sherrylone/m-mix)]
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204582292-eef1f644-f39f-46ae-98ba-313501bdb515.png" /></p>
* **SDMP**: Sucheng Ren, Huiyu Wang, Zhengqi Gao, Shengfeng He, Alan Yuille, Yuyin Zhou, Cihang Xie.
   - A Simple Data Mixing Prior for Improving Self-Supervised Learning. [[CVPR'2022](https://arxiv.org/abs/2206.07692)] [[code](https://github.com/oliverrensu/sdmp)]
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204585207-6fee3174-224d-44e6-bbd1-514d6697d128.png" /></p>
* **ScaleMix**: Xiao Wang, Haoqi Fan, Yuandong Tian, Daisuke Kihara, Xinlei Chen.
   - On the Importance of Asymmetry for Siamese Representation Learning. [[CVPR'2022](https://arxiv.org/abs/2204.00613)] [[code](https://github.com/facebookresearch/asym-siam)]
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204585304-03b6aa42-205c-4650-9d90-ecfc0928734e.png" /></p>
* **VLMixer**: Teng Wang, Wenhao Jiang, Zhichao Lu, Feng Zheng, Ran Cheng, Chengguo Yin, Ping Luo.
   - VLMixer: Unpaired Vision-Language Pre-training via Cross-Modal CutMix. [[ICML'2022](https://arxiv.org/abs/2206.08919)]
   <p align="center"><img width="60%" src="https://user-images.githubusercontent.com/44519745/204585605-f60ae9af-b7af-4af4-ac46-28bac51c7a02.png" /></p>
* **CropMix**: Junlin Han, Lars Petersson, Hongdong Li, Ian Reid.
   - CropMix: Sampling a Rich Input Distribution via Multi-Scale Cropping. [[ArXiv'2022](https://arxiv.org/abs/2205.15955)] [[code](https://github.com/JunlinHan/CropMix)]
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204585732-680295fe-4768-4199-af72-bda10edda644.png" /></p>

<p align="right">(<a href="#top">back to top</a>)</p>

## Mixup for Semi-supervised Learning

* **MixMatch**: David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, Colin Raffel.
   - MixMatch: A Holistic Approach to Semi-Supervised Learning. [[NIPS'2019](https://arxiv.org/abs/1905.02249)] [[code](https://github.com/google-research/mixmatch)]
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204580441-1fd71bf7-63f3-4935-9332-287642e0bcc8.png" /></p>
* **Pani VAT**: Ke Sun, Bing Yu, Zhouchen Lin, Zhanxing Zhu.
   - Patch-level Neighborhood Interpolation: A General and Effective Graph-based Regularization Strategy. [[ArXiv'2019](https://arxiv.org/abs/1911.09307)]
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204572993-8b3fa627-8c36-4763-a2a6-c9a90c5f0fc2.png" /></p>
* **ReMixMatch**: David Berthelot, dberth@google.com, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, Colin Raffel.
   - ReMixMatch: Semi-Supervised Learning with Distribution Matching and Augmentation Anchoring. [[ICLR'2020](https://openreview.net/forum?id=HklkeR4KPB)] [[code](https://github.com/google-research/remixmatch)]
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204579631-529bb505-a858-441f-9030-4a9b44273330.png" /></p>
* **Core-Tuning**: Yifan Zhang, Bryan Hooi, Dapeng Hu, Jian Liang, Jiashi Feng.
   - Unleashing the Power of Contrastive Self-Supervised Visual Models via Contrast-Regularized Fine-Tuning. [[NIPS'2021](https://arxiv.org/abs/2102.06605)] [[code](https://github.com/vanint/core-tuning)]
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204580135-ed6ba8b7-b69c-4683-90f0-9aa9cdd530bc.png" /></p>
* **MUM**: JongMok Kim, Jooyoung Jang, Seunghyeon Seo, Jisoo Jeong, Jongkeun Na, Nojun Kwak.
   - MUM : Mix Image Tiles and UnMix Feature Tiles for Semi-Supervised Object Detection. [[CVPR'2022](https://user-images.githubusercontent.com/44519745/225082975-4143e7f5-8873-433c-ab6f-6caa615f7120.png)] [[code](https://github.com/jongmokkim/mix-unmix)]
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/225082975-4143e7f5-8873-433c-ab6f-6caa615f7120.png" /></p>
* **DFixMatch**: Zicheng Liu, Siyuan Li, Ge Wang, Cheng Tan, Lirong Wu, Stan Z. Li.
   - Decoupled Mixup for Data-efficient Learning. [[Arxiv'2022](https://arxiv.org/abs/2203.10761)] [[code](https://github.com/Westlake-AI/openmixup)]
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204578387-4be9567c-963a-4d2d-8c1f-c7c5ade527b8.png" /></p>
* **LaserMix**: Lingdong Kong, Jiawei Ren, Liang Pan, Ziwei Liu.
   - LaserMix for Semi-Supervised LiDAR Semantic Segmentation. [[CVPR'2023](https://arxiv.org/abs/2207.00026)] [[code](https://github.com/ldkong1205/LaserMix)] [[project](https://ldkong.com/LaserMix)]
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/209255964-69cab84b-ae54-4e74-be4f-a23a836c665c.png" /></p>

<p align="right">(<a href="#top">back to top</a>)</p>

## Contribution

Feel free to send [pull requests](https://github.com/Westlake-AI/openmixup/pulls) to add more links with the following Markdown format. Notice that the Abbreviation, the code link, and the figure link are optional attributes. Current contributors include: Siyuan Li ([@Lupin1998](https://github.com/Lupin1998)) and Zicheng Liu ([@pone7](https://github.com/pone7)).

```markdown
* **Abbreviation**: Author List.
  - Paper Name. [[Conference'Year](link)] [[code](link)]
  <p align="center"><img width="90%" src="link_to_image" /></p>
```
