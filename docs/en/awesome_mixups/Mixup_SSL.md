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
  - [Related Project](#related-project)

## Mixup for Self-supervised Learning

* **MixCo: Mix-up Contrastive Learning for Visual Representation**<br>
*Sungnyun Kim, Gihun Lee, Sangmin Bae, Se-Young Yun*<br>
NIPSW'2020 [[Paper](https://arxiv.org/abs/2010.06300)]
[[Code](https://github.com/Lee-Gihun/MixCo-Mixup-Contrast)]
   <details close>
   <summary>MixCo Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204580767-c0730ac6-802f-40bc-92a8-4b7abe0acb99.png" /></p>
   </details>

* **Hard Negative Mixing for Contrastive Learning**<br>
*Yannis Kalantidis, Mert Bulent Sariyildiz, Noe Pion, Philippe Weinzaepfel, Diane Larlus*<br>
NIPS'2020 [[Paper](https://arxiv.org/abs/2010.01028)]
[[Code](https://europe.naverlabs.com/mochi)]
   <details close>
   <summary>MoCHi Framework</summary>
   <p align="center"><img width="40%" src="https://user-images.githubusercontent.com/44519745/204580935-80ebaaaa-2761-4da4-9b58-7755a0dc15c6.png" /></p>
   </details>

* **i-Mix A Domain-Agnostic Strategy for Contrastive Representation Learning**<br>
*Kibok Lee, Yian Zhu, Kihyuk Sohn, Chun-Liang Li, Jinwoo Shin, Honglak Lee*<br>
ICLR'2021 [[Paper](https://arxiv.org/abs/2010.08887)]
[[Code](https://github.com/kibok90/imix)]
   <details close>
   <summary>i-Mix Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204581084-5460bd37-4adb-4f01-b7af-fc88ceb2683e.png" /></p>
   </details>

* **Un-Mix: Rethinking Image Mixtures for Unsupervised Visual Representation**<br>
*Zhiqiang Shen, Zechun Liu, Zhuang Liu, Marios Savvides, Trevor Darrell, Eric Xing*<br>
AAAI'2022 [[Paper](https://arxiv.org/abs/2003.05438)]
[[Code](https://github.com/szq0214/Un-Mix)]
   <details close>
   <summary>Un-Mix Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204581434-dfbc11f8-e300-4dd7-bc84-adbe5a53dbf4.png" /></p>
   </details>

* **Beyond Single Instance Multi-view Unsupervised Representation Learning**<br>
*Xiangxiang Chu, Xiaohang Zhan, Xiaolin Wei*<br>
BMVC'2022 [[Paper](https://arxiv.org/abs/2011.13356)]
   <details close>
   <summary>BSIM Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204581834-3ead412b-359c-40ba-86ea-3ab54ead2c96.png" /></p>
   </details>

* **Improving Contrastive Learning by Visualizing Feature Transformation**<br>
*Rui Zhu, Bingchen Zhao, Jingen Liu, Zhenglong Sun, Chang Wen Chen*<br>
ICCV'2021 [[Paper](https://arxiv.org/abs/2108.02982)]
[[Code](https://github.com/DTennant/CL-Visualizing-Feature-Transformation)]
   <details close>
   <summary>FT Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204582514-426efca3-4e8b-48b8-b69a-ecebb94b7fa8.png" /></p>
   </details>

* **Piecing and Chipping: An effective solution for the information-erasing view generation in Self-supervised Learning**<br>
*Jingwei Liu, Yi Gu, Shentong Mo, Zhun Sun, Shumin Han, Jiafeng Guo, Xueqi Cheng*<br>
OpenReview'2021 [[Paper](https://openreview.net/forum?id=DnG8f7gweH4)]
   <details close>
   <summary>PCEA Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204582781-b54c4472-67eb-4e78-9362-44b990bbafa3.png" /></p>
   </details>

* **Contrast and Mix: Temporal Contrastive Video Domain Adaptation with Background Mixing**<br>
*Aadarsh Sahoo, Rutav Shah, Rameswar Panda, Kate Saenko, Abir Das*<br>
NIPS'2021 [[Paper](https://arxiv.org/abs/2011.02697)]
[[Code](https://cvir.github.io/projects/comix)]
   <details close>
   <summary>CoMix Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204583146-fc363695-6889-46cd-93f1-236bec9d5fb5.png" /></p>
   </details>

* **Boosting Discriminative Visual Representation Learning with Scenario-Agnostic Mixup**<br>
*Siyuan Li, Zicheng Liu, Di Wu, Zihan Liu, Stan Z. Li*<br>
Arxiv'2021 [[Paper](https://arxiv.org/abs/2111.15454)]
[[Code](https://github.com/Westlake-AI/openmixup)]
   <details close>
   <summary>SAMix Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/174272657-fb662377-b7c3-4faa-8d9b-ea6f1e08549e.png" /></p>
   </details>

* **MixSiam: A Mixture-based Approach to Self-supervised Representation Learning**<br>
*Xiaoyang Guo, Tianhao Zhao, Yutian Lin, Bo Du*<br>
OpenReview'2021 [[Paper](https://arxiv.org/abs/2111.02679)]
   <details close>
   <summary>MixSiam Framework</summary>
   <p align="center"><img width="60%" src="https://user-images.githubusercontent.com/44519745/204583448-8c1890fd-ce95-488a-9570-f7393a4d140a.png" /></p>
   </details>

* **Mix-up Self-Supervised Learning for Contrast-agnostic Applications**<br>
*Yichen Zhang, Yifang Yin, Ying Zhang, Roger Zimmermann*<br>
ICME'2021 [[Paper](https://arxiv.org/abs/2204.00901)]
   <details close>
   <summary>MixSSL Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204583625-58632669-05f0-445b-bd41-60aa37b515d4.png" /></p>
   </details>

* **Towards Domain-Agnostic Contrastive Learning**<br>
*Vikas Verma, Minh-Thang Luong, Kenji Kawaguchi, Hieu Pham, Quoc V. Le*<br>
ICML'2021 [[Paper](https://arxiv.org/abs/2011.04419)]
   <details close>
   <summary>DACL Framework</summary>
   <p align="center"><img width="50%" src="https://github.com/Westlake-AI/MogaNet/assets/44519745/19c8e3cb-db6f-463f-a765-c243b2f9e45a" /></p>
   </details>

* **Center-wise Local Image Mixture For Contrastive Representation Learning**<br>
*Hao Li, Xiaopeng Zhang, Hongkai Xiong*<br>
BMVC'2021 [[Paper](https://arxiv.org/abs/2011.02697)]
   <details close>
   <summary>CLIM Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204583930-15ab3916-9d8b-4adf-9a79-e40eabbbc255.png" /></p>
   </details>

* **Contrastive-mixup Learning for Improved Speaker Verification**<br>
*Xin Zhang, Minho Jin, Roger Cheng, Ruirui Li, Eunjung Han, Andreas Stolcke*<br>
ICASSP'2022 [[Paper](https://arxiv.org/abs/2202.10672)]
   <details close>
   <summary>Mixup Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204584821-5ef0cdd9-37bf-437a-a139-d21068452be1.png" /></p>
   </details>

* **ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning**<br>
*Jun Xia, Lirong Wu, Ge Wang, Jintao Chen, Stan Z.Li*<br>
ICML'2022 [[Paper](https://arxiv.org/abs/2110.02027)]
[[Code](https://github.com/junxia97/ProGCL)]
   <details close>
   <summary>ProGCL Framework</summary>
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/204584450-e31a6c9f-0f1c-4342-b907-1b80cae547ab.png" /></p>
   </details>

* **M-Mix: Generating Hard Negatives via Multi-sample Mixing for Contrastive Learning**<br>
*Shaofeng Zhang, Meng Liu, Junchi Yan, Hengrui Zhang, Lingxiao Huang, Pinyan Lu, Xiaokang Yang*<br>
KDD'2022 [[Paper](https://sherrylone.github.io/assets/KDD22_M-Mix.pdf)]
[[Code](https://github.com/Sherrylone/m-mix)]
   <details close>
   <summary>M-Mix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204582292-eef1f644-f39f-46ae-98ba-313501bdb515.png" /></p>
   </details>

* **A Simple Data Mixing Prior for Improving Self-Supervised Learning**<br>
*Sucheng Ren, Huiyu Wang, Zhengqi Gao, Shengfeng He, Alan Yuille, Yuyin Zhou, Cihang Xie*<br>
CVPR'2022 [[Paper](https://arxiv.org/abs/2206.07692)]
[[Code](https://github.com/oliverrensu/sdmp)]
   <details close>
   <summary>SDMP Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204585207-6fee3174-224d-44e6-bbd1-514d6697d128.png" /></p>
   </details>

* **On the Importance of Asymmetry for Siamese Representation Learning**<br>
*Xiao Wang, Haoqi Fan, Yuandong Tian, Daisuke Kihara, Xinlei Chen*<br>
CVPR'2022 [[Paper](https://arxiv.org/abs/2204.00613)]
[[Code](https://github.com/facebookresearch/asym-siam)]
   <details close>
   <summary>ScaleMix Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204585304-03b6aa42-205c-4650-9d90-ecfc0928734e.png" /></p>
   </details>

* **VLMixer: Unpaired Vision-Language Pre-training via Cross-Modal CutMix**<br>
*Teng Wang, Wenhao Jiang, Zhichao Lu, Feng Zheng, Ran Cheng, Chengguo Yin, Ping Luo*<br>
ICML'2022 [[Paper](https://arxiv.org/abs/2206.08919)]
   <details close>
   <summary>VLMixer Framework</summary>
   <p align="center"><img width="60%" src="https://user-images.githubusercontent.com/44519745/204585605-f60ae9af-b7af-4af4-ac46-28bac51c7a02.png" /></p>
   </details>

* **CropMix: Sampling a Rich Input Distribution via Multi-Scale Cropping**<br>
*Junlin Han, Lars Petersson, Hongdong Li, Ian Reid*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2205.15955)]
[[Code](https://github.com/JunlinHan/CropMix)]
   <details close>
   <summary>CropMix Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204585732-680295fe-4768-4199-af72-bda10edda644.png" /></p>
   </details>

* **- i-MAE: Are Latent Representations in Masked Autoencoders Linearly Separable**<br>
*Kevin Zhang, Zhiqiang Shen*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2210.11470)]
[[Code](https://github.com/vision-learning-acceleration-lab/i-mae)]
   <details close>
   <summary>i-MAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/211220785-5031f97c-c9a3-4ade-b344-48db01fc3760.png" /></p>
   </details>

* **MixMAE: Mixed and Masked Autoencoder for Efficient Pretraining of Hierarchical Vision Transformers**<br>
*Jihao Liu, Xin Huang, Jinliang Zheng, Yu Liu, Hongsheng Li*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2205.13137)]
[[Code](https://github.com/Sense-X/MixMIM)]
   <details close>
   <summary>MixMAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204315480-5c59ed60-7b5f-4da9-85fb-551a961fd731.png" /></p>
   </details>

* **Mixed Autoencoder for Self-supervised Visual Representation Learning**<br>
*Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2303.17152)]
   <details close>
   <summary>MixedAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/229929023-1ea53237-ebfb-4203-8b93-dd761d937b27.png" /></p>
   </details>

* **Inter-Instance Similarity Modeling for Contrastive Learning**<br>
*Chengchao Shen, Dawei Liu, Hao Tang, Zhe Qu, Jianxin Wang*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2306.12243)]
[[Code](https://github.com/visresearch/patchmix)]
   <details close>
   <summary>PatchMix Framework</summary>
   <p align="center"><img width="50%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/250166870-98280c28-2736-4f08-a418-d28e9ba3a588.png" /></p>
   </details>

<p align="right">(<a href="#top">back to top</a>)</p>

## Mixup for Semi-supervised Learning

* **MixMatch: A Holistic Approach to Semi-Supervised Learning**<br>
*David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, Colin Raffel*<br>
NIPS'2019 [[Paper](https://arxiv.org/abs/1905.02249)]
[[Code](https://github.com/google-research/mixmatch)]
   <details close>
   <summary>MixMatch Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204580441-1fd71bf7-63f3-4935-9332-287642e0bcc8.png" /></p>
   </details>

* **Patch-level Neighborhood Interpolation: A General and Effective Graph-based Regularization Strategy**<br>
*Ke Sun, Bing Yu, Zhouchen Lin, Zhanxing Zhu*<br>
ArXiv'2019 [[Paper](https://arxiv.org/abs/1911.09307)]
   <details close>
   <summary>Pani VAT Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204572993-8b3fa627-8c36-4763-a2a6-c9a90c5f0fc2.png" /></p>
   </details>

* **ReMixMatch: Semi-Supervised Learning with Distribution Matching and Augmentation Anchoring**<br>
*David Berthelot, dberth@google.com, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, Colin Raffel*<br>
ICLR'2020 [[Paper](https://openreview.net/forum?id=HklkeR4KPB)]
[[Code](https://github.com/google-research/remixmatch)]
   <details close>
   <summary>ReMixMatch Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204579631-529bb505-a858-441f-9030-4a9b44273330.png" /></p>
   </details>

* **DivideMix: Learning with Noisy Labels as Semi-supervised Learning**<br>
*Junnan Li, Richard Socher, Steven C.H. Hoi*<br>
ICLR'2020 [[Paper](https://arxiv.org/abs/2002.07394)]
[[Code](https://github.com/LiJunnan1992/DivideMix)]
   <details close>
   <summary>DivideMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/230495626-f0f3f52e-9f8a-472d-8ff2-b33356993e09.png" /></p>
   </details>

* **Unleashing the Power of Contrastive Self-Supervised Visual Models via Contrast-Regularized Fine-Tuning**<br>
*Yifan Zhang, Bryan Hooi, Dapeng Hu, Jian Liang, Jiashi Feng*<br>
NIPS'2021 [[Paper](https://arxiv.org/abs/2102.06605)]
[[Code](https://github.com/vanint/core-tuning)]
   <details close>
   <summary>Core-Tuning Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204580135-ed6ba8b7-b69c-4683-90f0-9aa9cdd530bc.png" /></p>
   </details>

* **MUM : Mix Image Tiles and UnMix Feature Tiles for Semi-Supervised Object Detection**<br>
*JongMok Kim, Jooyoung Jang, Seunghyeon Seo, Jisoo Jeong, Jongkeun Na, Nojun Kwak*<br>
CVPR'2022 [[Paper](https://user-images.githubusercontent.com/44519745/225082975-4143e7f5-8873-433c-ab6f-6caa615f7120.png)]
[[Code](https://github.com/jongmokkim/mix-unmix)]
   <details close>
   <summary>MUM Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/225082975-4143e7f5-8873-433c-ab6f-6caa615f7120.png" /></p>
   </details>

* **Decoupled Mixup for Data-efficient Learning**<br>
*Zicheng Liu, Siyuan Li, Ge Wang, Cheng Tan, Lirong Wu, Stan Z. Li*<br>
NIPS'2023 [[Paper](https://arxiv.org/abs/2203.10761)]
[[Code](https://github.com/Westlake-AI/openmixup)]
   <details close>
   <summary>DFixMatch Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204578387-4be9567c-963a-4d2d-8c1f-c7c5ade527b8.png" /></p>
   </details>

* **Manifold DivideMix: A Semi-Supervised Contrastive Learning Framework for Severe Label Noise**<br>
*Fahimeh Fooladgar, Minh Nguyen Nhat To, Parvin Mousavi, Purang Abolmaesumi*<br>
Arxiv'2023 [[Paper](https://arxiv.org/abs/2308.06861)]
[[Code](https://github.com/Fahim-F/ManifoldDivideMix)]
   <details close>
   <summary>MixEMatch Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/268411562-4263ccd5-a31c-4020-9281-ba4bc3d9fc54.png" /></p>
   </details>

* **LaserMix for Semi-Supervised LiDAR Semantic Segmentation**<br>
*Lingdong Kong, Jiawei Ren, Liang Pan, Ziwei Liu*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2207.00026)]
[[Code](https://github.com/ldkong1205/LaserMix)] [[project](https://ldkong.com/LaserMix)]
   <details close>
   <summary>LaserMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/209255964-69cab84b-ae54-4e74-be4f-a23a836c665c.png" /></p>
   </details>

* **Dual-Decoder Consistency via Pseudo-Labels Guided Data Augmentation for Semi-Supervised Medical Image Segmentation**<br>
*Yuanbin Chen, Tao Wang, Hui Tang, Longxuan Zhao, Ruige Zong, Tao Tan, Xinlin Zhang, Tong Tong*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2308.16573)]
   <details close>
   <summary>DCPA Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/268410560-a45c03d9-beb1-4b74-a34b-4d1ecd356de9.png" /></p>
   </details>

<p align="right">(<a href="#top">back to top</a>)</p>

## Contribution

Feel free to send [pull requests](https://github.com/Westlake-AI/openmixup/pulls) to add more links with the following Markdown format. Notice that the Abbreviation, the code link, and the figure link are optional attributes. Current contributors include: Siyuan Li ([@Lupin1998](https://github.com/Lupin1998)) and Zicheng Liu ([@pone7](https://github.com/pone7)).

```markdown
* **TITLE**<br>
*AUTHER*<br>
PUBLISH'YEAR [[Paper](link)] [[Code](link)]
   <details close>
   <summary>ABBREVIATION Framework</summary>
   <p align="center"><img width="90%" src="link_to_image" /></p>
   </details>
```

## Related Project

- [Awesome-Mixup](https://github.com/Westlake-AI/Awesome-Mixup): Awesome List of Mixup Augmentation Papers for Visual Representation Learning.
- [Awesome-Mix](https://github.com/ChengtaiCao/Awesome-Mix): An awesome list of papers for `A Survey of Mix-based Data Augmentation: Taxonomy, Methods, Applications, and Explainability, we categorize them based on our proposed taxonomy`.
- [survery-image-mixing-and-deleting-for-data-augmentation](https://github.com/humza909/survery-image-mixing-and-deleting-for-data-augmentation): An awesome list of papers for `Survey: Image Mixing and Deleting for Data Augmentation`.
- [awesome-mixup](https://github.com/demoleiwang/awesome-mixup): A collection of awesome papers about mixup.
- [awesome-mixed-sample-data-augmentation](https://github.com/JasonZhang156/awesome-mixed-sample-data-augmentation): A collection of awesome things about mixed sample data augmentation.
- [data-augmentation-review](https://github.com/AgaMiko/data-augmentation-review): List of useful data augmentation resources.
