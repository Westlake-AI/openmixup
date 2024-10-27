# Awesome Masked Image Modeling for Visual Represention

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
<!-- ![GitHub stars](https://img.shields.io/github/stars/Westlake-AI/openmixup?color=blue) ![GitHub forks](https://img.shields.io/github/forks/Westlake-AI/openmixup?color=yellow&label=Fork) -->

**We summarize masked image modeling (MIM) methods proposed for self-supervised visual representation learning.**
The list of awesome MIM methods is summarized in chronological order and is on updating.

* To find related papers and their relationships, check out [Connected Papers](https://www.connectedpapers.com/), which visualizes the academic field in a graph representation.
* To export BibTeX citations of papers, check out [ArXiv](https://arxiv.org/) or [Semantic Scholar](https://www.semanticscholar.org/) of the paper for professional reference formats.

## Table of Contents

- [Awesome Masked Modeling for Self-supervised Vision Represention and Beyond](#awesome-masked-modeling-for-self-supervised-vision-represention-and-beyond)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Fundamental MIM Methods](#fundamental-mim-methods)
    - [MIM for Transformers](#mim-for-transformers)
    - [MIM with Constrastive Learning](#mim-with-constrastive-learning)
    - [MIM for Transformers and CNNs](#mim-for-transformers-and-cnns)
    - [MIM with Advanced Masking](#mim-with-advanced-masking)
    - [MIM for Multi-Modality](#mim-for-multi-modality)
    - [MIM for Vision Generalist Model](#mim-for-vision-generalist-model)
    - [Image Generation](#image-generation)
  - [MIM for CV Downstream Tasks](#mim-for-cv-downstream-tasks)
    - [Object Detection](#object-detection)
    - [Video Rrepresentation](#video-rrepresentation)
    - [Knowledge Distillation and Few-shot Classification](#knowledge-distillation-and-few-shot-classification)
    - [Efficient Fine-tuning](#efficient-fine-tuning)
    - [Medical Image](#medical-image)
    - [Face Recognition](#face-recognition)
    - [Scene Text Recognition (OCR)](#scene-text-recognition-ocr)
    - [Remote Sensing Image](#remote-sensing-image)
    - [3D Representation Learning](#3d-representation-learning)
    - [Depth Estimation](#depth-estimation)
  - [Audio and Speech](#audio-and-speech)
  - [AI for Science](#ai-for-science)
    - [Protein](#protein)
    - [Chemistry](#chemistry)
    - [Physics](#physics)
  - [Neuroscience Learning](#time-series-and-neuroscience-learning)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Analysis and Understanding of Masked Modeling](#analysis-and-understanding-of-masked-modeling)
  - [Survey](#survey)
  - [Contribution](#contribution)
  - [Related Project](#related-project)
    - [Paper List of Masked Image Modeling](#paper-list-of-masked-image-modeling)
    - [Project of Self-supervised Learning](#project-of-self-supervised-learning)

## Fundamental MIM Methods

<p align="center" width="100%">
  <img src='https://github.com/Lupin1998/Awesome-MIM/assets/44519745/ec3f11eb-b12d-4ebc-a129-fc951018ddcd' width="90%">
</p>
The overview of the basic MIM framework, containing four building blocks with their internal components and functionalities. All MIM research can be summarized as innovations upon these four blocks, i.e., Masking, Encoder, Target, and Head. Frameworks of masked modeling in other modalities are similar to this framework.

### MIM for Transformers

* **Generative Pretraining from Pixels**<br>
*Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, David Luan, Ilya Sutskever*<br>
ICML'2020 [[Paper](http://proceedings.mlr.press/v119/chen20s/chen20s.pdf)]
[[Code](https://github.com/openai/image-gpt)]
   <details close>
   <summary>iGPT Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204300433-a0b6b25b-9f6f-431b-bbfd-19169d8cbca6.png" /></p>
   </details>

* **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**<br>
*Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby*<br>
ICLR'2021 [[Paper](https://arxiv.org/abs/2010.11929)]
[[Code](https://github.com/google-research/vision_transformer)]
   <details close>
   <summary>ViT Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204301490-5673cc4c-93d1-435d-a266-ec5a0294bf3b.png" /></p>
   </details>

* **BEiT: BERT Pre-Training of Image Transformers**<br>
*Hangbo Bao, Li Dong, Furu Wei*<br>
ICLR'2022 [[Paper](https://arxiv.org/abs/2106.08254)]
[[Code](https://github.com/microsoft/unilm/tree/master/beit)]
   <details close>
   <summary>BEiT Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204301720-156e15e1-a00a-4946-b17f-d2620d2be3d6.png" /></p>
   </details>

* **iBOT: Image BERT Pre-Training with Online Tokenizer**<br>
*Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, Tao Kong*<br>
ICLR'2022 [[Paper](https://arxiv.org/abs/2111.07832)]
[[Code](https://github.com/bytedance/ibot)]
   <details close>
   <summary>iBOT Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204301946-1b18e2a8-b205-4d85-8ea9-bd2e4d529a70.png" /></p>
   </details>

* **Masked Autoencoders Are Scalable Vision Learners**<br>
*Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick*<br>
CVPR'2022 [[Paper](https://arxiv.org/abs/2111.06377)]
[[Code](https://github.com/facebookresearch/mae)]
   <details close>
   <summary>MAE Framework</summary>
   <p align="center"><img width="65%" src="https://user-images.githubusercontent.com/44519745/204302185-1b854627-597b-416a-aa85-23dc6c87b59e.png" /></p>
   </details>

* **SimMIM: A Simple Framework for Masked Image Modeling**<br>
*Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, Han Hu*<br>
CVPR'2022 [[Paper](https://arxiv.org/abs/2111.09886)]
[[Code](https://github.com/microsoft/simmim)]
   <details close>
   <summary>SimMIM Framework</summary>
   <p align="center"><img width="65%" src="https://user-images.githubusercontent.com/44519745/204302529-8075a5cc-a2e8-4245-891b-8f74c1bc1734.png" /></p>
   </details>

* **Masked Feature Prediction for Self-Supervised Visual Pre-Training**<br>
*Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, Christoph Feichtenhofer*<br>
CVPR'2022 [[Paper](https://arxiv.org/abs/2112.09133)]
[[Code](https://github.com/facebookresearch/SlowFast)]
   <details close>
   <summary>MaskFeat Framework</summary>
   <p align="center"><img width="60%" src="https://user-images.githubusercontent.com/44519745/204302699-10f1f1d4-2bb4-428a-b43b-2972ba915286.png" /></p>
   </details>

* **data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language**<br>
*Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli*<br>
ICML'2022 [[Paper](https://arxiv.org/abs/2202.03555)]
[[Code](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)]
   <details close>
   <summary>data2vec Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204302962-e44f4eed-b7d0-4b64-8696-ae6f349400fb.png" /></p>
   </details>

* **Position Prediction as an Effective Pretraining Strategy**<br>
*Shuangfei Zhai, Navdeep Jaitly, Jason Ramapuram, Dan Busbridge, Tatiana Likhomanenko, Joseph Yitan Cheng, Walter Talbott, Chen Huang, Hanlin Goh, Joshua Susskind*<br>
ICML'2022 [[Paper](https://arxiv.org/abs/2207.07611)]
   <details close>
   <summary>MP3 Framework</summary>
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/206919419-aa867bf6-f729-4bf7-8f70-c21f17bf8cec.png" /></p>
   </details>

* **PeCo: Perceptual Codebook for BERT Pre-training of Vision Transformers**<br>
*Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu*<br>
AAAI'2023 [[Paper](https://arxiv.org/abs/2111.12710)]
[[Code](https://github.com/microsoft/PeCo)]
   <details close>
   <summary>PeCo Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/265825505-92b2c2d9-0120-4c65-af9e-528249555d87.png" /></p>
   </details>

* **MC-SSL0.0: Towards Multi-Concept Self-Supervised Learning**<br>
*Sara Atito, Muhammad Awais, Ammarah Farooq, Zhenhua Feng, Josef Kittler*<br>
ArXiv'2021 [[Paper](https://arxiv.org/abs/2111.15340)]
   <details close>
   <summary>MC-SSL0.0 Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204303461-5d4fb0a1-bff5-4f0a-afea-ee0d81ce17ba.png" /></p>
   </details>

* **mc-BEiT: Multi-choice Discretization for Image BERT Pre-training**<br>
*Xiaotong Li, Yixiao Ge, Kun Yi, Zixuan Hu, Ying Shan, Ling-Yu Duan*<br>
ECCV'2022 [[Paper](https://arxiv.org/abs/2203.15371)]
[[Code](https://github.com/lixiaotong97/mc-BEiT)]
   <details close>
   <summary>mc-BEiT Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204304102-a17e3bb2-ffe5-4b42-bf5f-4dece8809391.png" /></p>
   </details>

* **Bootstrapped Masked Autoencoders for Vision BERT Pretraining**<br>
*Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu*<br>
ECCV'2022 [[Paper](https://arxiv.org/abs/2207.07116)]
[[Code](https://github.com/LightDXY/BootMAE)]
   <details close>
   <summary>BootMAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204304317-0cbe6647-769b-4737-9f65-85c4e47ec944.png" /></p>
   </details>

* **SdAE: Self-distillated Masked Autoencoder**<br>
*Yabo Chen, Yuchen Liu, Dongsheng Jiang, Xiaopeng Zhang, Wenrui Dai, Hongkai Xiong, Qi Tian*<br>
ECCV'2022 [[Paper](https://arxiv.org/abs/2208.00449)]
[[Code](https://github.com/AbrahamYabo/SdAE)]
   <details close>
   <summary>SdAE Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204304730-edc6fe19-b12a-4986-922a-9694230e9ef2.png" /></p>
   </details>

* **MultiMAE: Multi-modal Multi-task Masked Autoencoders**<br>
*Roman Bachmann, David Mizrahi, Andrei Atanov, Amir Zamir*<br>
ECCV'2022 [[Paper](https://arxiv.org/abs/2204.01678)]
[[Code](https://github.com/EPFL-VILAB/MultiMAE)]
   <details close>
   <summary>MultiMAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204304575-577cc0f0-3ac7-4f02-b884-48ec8d061476.png" /></p>
   </details>

* **SupMAE: Supervised Masked Autoencoders Are Efficient Vision Learners**<br>
*Feng Liang, Yangguang Li, Diana Marculescu*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2205.14540)]
[[Code](https://github.com/cmu-enyac/supmae)]
   <details close>
   <summary>SupMAE Framework</summary>
   <p align="center"><img width="65%" src="https://user-images.githubusercontent.com/44519745/204305079-c99782a4-ba5f-4785-a2c6-4990da13a493.png" /></p>
   </details>

* **MVP: Multimodality-guided Visual Pre-training**<br>
*Longhui Wei, Lingxi Xie, Wengang Zhou, Houqiang Li, Qi Tian*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2203.05175)]
   <details close>
   <summary>MVP Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204305240-24cee1ff-13d6-4936-93a1-8a4614faed99.png" /></p>
   </details>

* **The Devil is in the Frequency: Geminated Gestalt Autoencoder for Self-Supervised Visual Pre-Training**<br>
*Hao Liu, Xinghua Jiang, Xin Li, Antai Guo, Deqiang Jiang, Bo Ren*<br>
AAAI'2023 [[Paper](https://arxiv.org/abs/2204.08227)]
   <details close>
   <summary>Ge2AE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204305475-c462edf5-6e20-4f43-a1e1-f06641d13966.png" /></p>
   </details>

* **ConvMAE: Masked Convolution Meets Masked Autoencoders**<br>
*Peng Gao, Teli Ma, Hongsheng Li, Ziyi Lin, Jifeng Dai, Yu Qiao*<br>
NeurIPS'2022 [[Paper](https://arxiv.org/abs/2205.03892)]
[[Code](https://github.com/Alpha-VL/ConvMAE)]
   <details close>
   <summary>ConvMAE Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204305687-8f04d9f7-dc60-4ff0-8f94-5e72795774ca.png" /></p>
   </details>

* **Mimic before Reconstruct: Enhancing Masked Autoencoders with Feature Mimicking**<br>
*Peng Gao, Renrui Zhang, Rongyao Fang, Ziyi Lin, Hongyang Li, Hongsheng Li, Qiao Yu*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2303.05475)]
[[Code](https://github.com/alpha-vl/convmae)]
   <details close>
   <summary>MR-MAE (ConvMAE.V2) Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/236312228-1038fc6c-af93-46f2-b0aa-6f121f8388be.png" /></p>
   </details>

* **Green Hierarchical Vision Transformer for Masked Image Modeling**<br>
*Lang Huang, Shan You, Mingkai Zheng, Fei Wang, Chen Qian, Toshihiko Yamasaki*<br>
NeurIPS'2022 [[Paper](https://arxiv.org/abs/2205.13515)]
[[Code](https://github.com/LayneH/GreenMIM)]
   <details close>
   <summary>GreenMIM Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204305942-b22b5064-26d9-4f0b-9a2a-88012873f4fa.png" /></p>
   </details>

* **Test-Time Training with Masked Autoencoders**<br>
*Yossi Gandelsman, Yu Sun, Xinlei Chen, Alexei A. Efros*<br>
NeurIPS'2022 [[Paper](https://arxiv.org/abs/2209.07522)]
[[Code](https://github.com/yossigandelsman/test_time_training_mae)]
   <details close>
   <summary>TTT-MAE Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204306169-63fd5383-ee33-47f0-a955-971cfbd150ae.png" /></p>
   </details>

* **HiViT: Hierarchical Vision Transformer Meets Masked Image Modeling**<br>
*Xiaosong Zhang, Yunjie Tian, Wei Huang, Qixiang Ye, Qi Dai, Lingxi Xie, Qi Tian*<br>
ICLR'2023 [[Paper](https://arxiv.org/abs/2205.14949)]
   <details close>
   <summary>HiViT Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204306642-4764b620-0a1d-4625-8f22-e0fbcc3f5b2e.png" /></p>
   </details>

* **Contrastive Learning Rivals Masked Image Modeling in Fine-tuning via Feature Distillation**<br>
*Yixuan Wei, Han Hu, Zhenda Xie, Zheng Zhang, Yue Cao, Jianmin Bao, Dong Chen, Baining Guo*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2205.14141)]
[[Code](https://github.com/SwinTransformer/Feature-Distillation)]
   <details close>
   <summary>FD Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204306965-8c8ccfd7-353d-431a-819b-9872cc95bf9b.png" /></p>
   </details>

* **Object-wise Masked Autoencoders for Fast Pre-training**<br>
*Jiantao Wu, Shentong Mo*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2205.14338)]
   <details close>
   <summary>ObjMAE Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204307186-fcca6049-8ed6-4010-967b-b7cf93bd0619.png" /></p>
   </details>

* **Efficient Self-supervised Vision Pretraining with Local Masked Reconstruction**<br>
*Jun Chen, Ming Hu, Boyang Li, Mohamed Elhoseiny*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2206.00790)]
[[Code](https://github.com/junchen14/LoMaR)]
   <details close>
   <summary>LoMaR Framework</summary>
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/204307493-233177c7-3a18-4228-9d3b-34678dee8fe3.png" /></p>
   </details>

* **Extreme Masking for Learning Instance and Distributed Visual Representations**<br>
*Zhirong Wu, Zihang Lai, Xiao Sun, Stephen Lin*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2206.04667)]
   <details close>
   <summary>ExtreMA Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204307667-973af3e2-e50a-4cf7-9a31-035668aed4e3.png" /></p>
   </details>

* **BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers**<br>
*Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, Furu Wei*<br>
ArXiv'2022 [[Paper](http://arxiv.org/abs/2208.06366)]
[[Code](https://aka.ms/beit)]
   <details close>
   <summary>BEiT.V2 Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204307878-40ba7e59-2894-4a8e-93d5-8a8d43f12744.png" /></p>
   </details>

* **MILAN: Masked Image Pretraining on Language Assisted Representation**<br>
*Zejiang Hou, Fei Sun, Yen-Kuang Chen, Yuan Xie, Sun-Yuan Kung*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2208.06049)]
[[Code](https://github.com/zejiangh/milan)]
   <details close>
   <summary>MILAN Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204308146-edae9cfb-3f03-4b13-bf11-51620ebc945d.png" /></p>
   </details>

* **Exploring The Role of Mean Teachers in Self-supervised Masked Auto-Encoders**<br>
*Youngwan Lee, Jeffrey Willette, Jonghee Kim, Juho Lee, Sung Ju Hwang*<br>
ICLR'2023 [[Paper](https://arxiv.org/abs/2210.02077)]
   <details close>
   <summary>RC-MAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204310107-ff38a657-9fac-4271-89a2-e28a2805bf5a.png" /></p>
   </details>

* **Denoising Masked AutoEncoders are Certifiable Robust Vision Learners**<br>
*Quanlin Wu, Hang Ye, Yuntian Gu, Huishuai Zhang, Liwei Wang, Di He*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2210.06983)]
[[Code](https://github.com/quanlin-wu/dmae)]
   <details close>
   <summary>DMAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204310334-f2bb8c49-d2a5-4017-9501-b0bd76340bdc.png" /></p>
   </details>

* **A Unified View of Masked Image Modeling**<br>
*Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, Furu Wei*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2210.10615)]
[[Code](https://aka.ms/unimim)]
   <details close>
   <summary>MaskDistill Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204310534-7c1bf6fc-690b-4dd3-889d-4488b8a892ea.png" /></p>
   </details>

* **DILEMMA: Self-Supervised Shape and Texture Learning with Transformers**<br>
*Sepehr Sameni, Simon Jenni, Paolo Favaro*<br>
AAAI'2023 [[Paper](https://arxiv.org/abs/2204.04788)]
   <details close>
   <summary>DILEMMA Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/206920238-6f520585-e9c1-4e7a-89eb-3d9379931279.png" /></p>
   </details>

* **i-MAE: Are Latent Representations in Masked Autoencoders Linearly Separable**<br>
*Kevin Zhang, Zhiqiang Shen*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2210.11470)]
[[Code](https://github.com/vision-learning-acceleration-lab/i-mae)]
   <details close>
   <summary>i-MAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/211220785-5031f97c-c9a3-4ade-b344-48db01fc3760.png" /></p>
   </details>

* **EVA: Exploring the Limits of Masked Visual Representation Learning at Scale**<br>
*Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, Yue Cao*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2211.07636)]
[[Code](https://github.com/baaivision/EVA)]
   <details close>
   <summary>EVA Framework</summary>
   <p align="center"><img width="60%" src="https://user-images.githubusercontent.com/44519745/206920442-4d896aca-1765-4e66-9afb-c76017bc3521.png" /></p>
   </details>

* **EVA-02: A Visual Representation for Neon Genesis**<br>
*Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, Yue Cao*<br>
CVPR'2024 [[Paper](https://arxiv.org/abs/2303.11331)]
[[Code](https://github.com/baaivision/EVA/tree/master/EVA-02)]
   <details close>
   <summary>EVA-02 Framework</summary>
   <p align="center"><img width="50%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/0dc8f561-dd10-4950-8472-3b7f21210c82" /></p>
   </details>

* **Context Autoencoder for Self-Supervised Representation Learning**<br>
*Xiaokang Chen, Mingyu Ding, Xiaodi Wang, Ying Xin, Shentong Mo, Yunhao Wang, Shumin Han, Ping Luo, Gang Zeng, Jingdong Wang*<br>
IJCV'2023 [[Paper](https://arxiv.org/abs/2202.03026)]
[[Code](https://github.com/lxtGH/CAE)]
   <details close>
   <summary>CAE Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/234667973-6f98f65e-662c-4934-be85-efa60f3fc20a.png" /></p>
   </details>

* **CAE v2: Context Autoencoder with CLIP Target**<br>
*Xinyu Zhang, Jiahui Chen, Junkun Yuan, Qiang Chen, Jian Wang, Xiaodi Wang, Shumin Han, Xiaokang Chen, Jimin Pi, Kun Yao, Junyu Han, Errui Ding, Jingdong Wang*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2211.09799)]
   <details close>
   <summary>CAE.V2 Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/206920593-c703518b-47f9-4f61-a319-5ba0099c902d.png" /></p>
   </details>

* **FastMIM: Expediting Masked Image Modeling Pre-training for Vision**<br>
*Jianyuan Guo, Kai Han, Han Wu, Yehui Tang, Yunhe Wang, Chang Xu*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2212.06593)]
   <details close>
   <summary>FastMIM Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/210276245-83f9b564-2bdb-48b7-922c-dc36e3d5c20f.png" /></p>
   </details>

* **Exploring Target Representations for Masked Autoencoders**<br>
*Xingbin Liu, Jinghao Zhou, Tao Kong, Xianming Lin, Rongrong Ji*<br>
ICLR'2024 [[Paper](https://arxiv.org/abs/2209.03917)]
[[Code](https://github.com/liuxingbin/dbot)]
   <details close>
   <summary>dBOT Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/225108834-36affd27-fbae-46f0-92ca-d5a35a39023d.png" /></p>
   </details>

* **Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language**<br>
*Alexei Baevski, Arun Babu, Wei-Ning Hsu, and Michael Auli*<br>
ICML'2023 [[Paper](https://arxiv.org/abs/2212.07525)]
[[Code](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)]
   <details close>
   <summary>Data2Vec.V2 Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/207722013-4fc539f7-3d45-4eb8-8037-c4fa210738d6.png" /></p>
   </details>

* **Masked autoencoders are effective solution to transformer data-hungry**<br>
*Jiawei Mao, Honggu Zhou, Xuesong Yin, Yuanqi Chang. Binling Nie. Rui Xu*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2212.05677)]
[[Code](https://github.com/Talented-Q/SDMAE)]
   <details close>
   <summary>SDMAE Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/211220908-70f4c587-80a9-4427-8a68-b17593de8b0a.png" /></p>
   </details>

* **TinyMIM: An Empirical Study of Distilling MIM Pre-trained Models**<br>
*Sucheng Ren, Fangyun Wei, Zheng Zhang, Han Hu*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2301.01296)]
[[Code](https://github.com/OliverRensu/TinyMIM)]
   <details close>
   <summary>TinyMIM Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/210646611-f5022c04-9c34-465c-b4dd-53ca097f47d8.png" /></p>
   </details>

* **Disjoint Masking with Joint Distillation for Efficient Masked Image Modeling**<br>
*Xin Ma, Chang Liu, Chunyu Xie, Long Ye, Yafeng Deng, Xiangyang Ji*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2301.00230)]
[[Code](https://github.com/mx-mark/dmjd)]
   <details close>
   <summary>DMJD Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/210645728-7a066877-9eea-4fdb-a13a-df6863a287e6.png" /></p>
   </details>

* **Mixed Autoencoder for Self-supervised Visual Representation Learning**<br>
*Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2303.17152)]
   <details close>
   <summary>MixedAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/229929023-1ea53237-ebfb-4203-8b93-dd761d937b27.png" /></p>
   </details>

* **Masked Image Modeling with Local Multi-Scale Reconstruction**<br>
*Haoqing Wang, Yehui Tang, Yunhe Wang, Jianyuan Guo, Zhi-Hong Deng, Kai Han*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2303.05251)]
[[Code](https://github.com/Haoqing-Wang/LocalMIM)]
   <details close>
   <summary>LocalMAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/233732370-4ad8b717-5a86-4957-8d8a-494cc9198685.png" /></p>
   </details>

* **Stare at What You See: Masked Image Modeling without Reconstruction**<br>
*Hongwei Xue, Peng Gao, Hongyang Li, Yu Qiao, Hao Sun, Houqiang Li, Jiebo Luo*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2211.08887)]
[[Code](https://github.com/OpenPerceptionX/maskalign)]
   <details close>
   <summary>MaskAlign Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/236316028-df4132f1-2e76-4cef-8f88-ce1d3e84b127.png" /></p>
   </details>

* **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**<br>
*Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2301.08243)]
   <details close>
   <summary>I-JEPA Framework</summary>
   <p align="center"><img width="55%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/245952605-af5cfbb9-4cba-4bdc-803a-2e7022dd4ed1.png" /></p>
   </details>

* **MOMA: Distill from Self-Supervised Teachers**<br>
*Yuchong Yao, Nandakishor Desai, Marimuthu Palaniswami*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2302.02089)]
   <details close>
   <summary>MOMA Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/236316583-28639d42-3574-4dcc-8bd4-da377cce29a4.png" /></p>
   </details>

* **PixMIM: Rethinking Pixel Reconstruction in Masked Image Modeling**<br>
*Yuan Liu, Songyang Zhang, Jiacheng Chen, Kai Chen, Dahua Lin*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2303.02416)]
[[Code](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/pixmim)]
   <details close>
   <summary>PixMIM Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/236320141-75a1c36e-ab6e-4bbe-bc46-b94f91db1439.png" /></p>
   </details>

* **Img2Vec: A Teacher of High Token-Diversity Helps Masked AutoEncoders**<br>
*Heng Pan, Chenyang Liu, Wenxiao Wang, Li Yuan, Hongfa Wang, Zhifeng Li, Wei Liu*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2304.12535)]
   <details close>
   <summary>Img2Vec Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/236318192-b39c9900-db30-4e18-bb1b-a5020723b906.png" /></p>
   </details>

* **A Closer Look at Self-Supervised Lightweight Vision Transformers**<br>
*Shaoru Wang, Jin Gao, Zeming Li, Xiaoqin Zhang, Weiming Hu*<br>
ICML'2023 [[Paper](https://arxiv.org/abs/2205.14443)]
[[Code](https://github.com/wangsr126/mae-lite)]
   <details close>
   <summary>MAE-Lite Framework</summary>
   <p align="center"><img width="80%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/218d8909-9ad5-43af-9eeb-42c9e0aea9ed" /></p>
   </details>

* **Architecture-Agnostic Masked Image Modeling - From ViT back to CNN**<br>
*Siyuan Li, Di Wu, Fang Wu, Zelin Zang, Stan.Z.Li*<br>
ICML'2023 [[Paper](https://arxiv.org/abs/2205.13943)]
[[Code](https://github.com/Westlake-AI/openmixup)] [[project](https://github.com/Westlake-AI/A2MIM)]
   <details close>
   <summary>A2MIM Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204314681-d953cffc-8ba7-481c-925e-c89084f83c56.png" /></p>
   </details>

* **Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles**<br>
*Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan, Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed, Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer*<br>
ICML'2023 [[Paper](https://arxiv.org/abs/2306.00989)]
[[Code](https://github.com/facebookresearch/hiera)]
   <details close>
   <summary>Hiera Framework</summary>
   <p align="center"><img width="95%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/f4a007c0-4479-4701-875d-b182bf963332" /></p>
   </details>

* **The effectiveness of MAE pre-pretraining for billion-scale pretraining**<br>
*Mannat Singh, Quentin Duval, Kalyan Vasudev Alwala, Haoqi Fan, Vaibhav Aggarwal, Aaron Adcock, Armand Joulin, Piotr Dollár, Christoph Feichtenhofer, Ross Girshick, Rohit Girdhar, Ishan Misra*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2303.13496)]
   <details close>
   <summary>WSP Framework</summary>
   <p align="center"><img width="50%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/59a0eb5c-5da6-4821-83fe-bba7567c3e6f" /></p>
   </details>

* **Learning to Mask and Permute Visual Tokens for Vision Transformer Pre-Training**<br>
*Lorenzo Baraldi, Roberto Amoroso, Marcella Cornia, Lorenzo Baraldi, Andrea Pilzer, Rita Cucchiara*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2306.07346)]
[[Code](https://github.com/aimagelab/mapet)]
   <details close>
   <summary>MaPeT Framework</summary>
   <p align="center"><img width="95%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/71967291-abc1-4136-b03a-e05cf7b6f7ef" /></p>
   </details>

* **BIM: Block-Wise Self-Supervised Learning with Masked Image Modeling**<br>
*Yixuan Luo, Mengye Ren, Sai Qian Zhang*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2311.17218)]

* **R-MAE: Regions Meet Masked Autoencoders**<br>
*Duy-Kien Nguyen, Vaibhav Aggarwal, Yanghao Li, Martin R. Oswald, Alexander Kirillov, Cees G. M. Snoek, Xinlei Chen*<br>
ICLR'2024 [[Paper](https://arxiv.org/abs/2306.05411)]
[[Code](https://github.com/facebookresearch/r-mae)]
   <details close>
   <summary>R-MAE Framework</summary>
   <p align="center"><img width="55%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/0c8747ed-477b-4217-b84e-cdf1a983bdbf" /></p>
   </details>

* **Improving Pixel-based MIM by Reducing Wasted Modeling Capability**<br>
*Yuan Liu, Songyang Zhang, Jiacheng Chen, Zhaohui Yu, Kai Chen, Dahua Lin*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2308.00261)]
[[Code](https://github.com/open-mmlab/mmpretrain/tree/dev)]
   <details close>
   <summary>MFM Framework</summary>
   <p align="center"><img width="80%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/263502915-4f9f94c8-3746-4745-80ab-a16e7eed2f5c.png" /></p>
   </details>

* **SparseMAE: Sparse Training Meets Masked Autoencoders**<br>
*Aojun Zhou, Yang Li, Zipeng Qin, Jianbo Liu, Junting Pan, Renrui Zhang, Rui Zhao, Peng Gao, Hongsheng Li*<br>
ICCV'2023 [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_SparseMAE_Sparse_Training_Meets_Masked_Autoencoders_ICCV_2023_paper.pdf)]
[[Code](https://github.com/aojunzz/SparseMAE)]
   <details close>
   <summary>SparseMAE Framework</summary>
   <p align="center"><img width="80%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273472384-7e1b04b9-03a3-44ee-a821-d959ba26e820.png" /></p>
   </details>

* **Improving Adversarial Robustness of Masked Autoencoders via Test-time Frequency-domain Prompting**<br>
*Qidong Huang, Xiaoyi Dong, Dongdong Chen, Yinpeng Chen, Lu Yuan, Gang Hua, Weiming Zhang, Nenghai Yu*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2308.10315)]
[[Code](https://github.com/shikiw/RobustMAE)]
   <details close>
   <summary>RobustMAE Framework</summary>
   <p align="center"><img width="50%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273511030-25d49434-d197-42fa-a11a-2ce02458b938.png" /></p>
   </details>

* **DeepMIM: Deep Supervision for Masked Image Modeling**<br>
*Sucheng Ren, Fangyun Wei, Samuel Albanie, Zheng Zhang, Han Hu*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2303.08817)]
[[Code](https://github.com/oliverrensu/deepmim)]
   <details close>
   <summary>RobustMAE Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/5df9e2dd-ee9f-4771-b509-31dfc1593c24" /></p>
   </details>

* **Rethinking Patch Dependence for Masked Autoencoders**<br>
*Letian Fu, Long Lian, Renhao Wang, Baifeng Shi, Xudong Wang, Adam Yala, Trevor Darrell, Alexei A. Efros, Ken Goldberg*<br>
ArXiv'2024 [[Paper](https://arxiv.org/abs/2401.14391)]
   <details close>
   <summary>CrossMAE Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/a64fc186-1082-40ba-bfe9-811c4a4cfe15)" /></p>
   </details>

* **Deconstructing Denoising Diffusion Models for Self-Supervised Learning**<br>
*Xinlei Chen, Zhuang Liu, Saining Xie, Kaiming He*<br>
ArXiv'2024 [[Paper](https://arxiv.org/abs/2401.14404)]
   <details close>
   <summary>l-DAE Framework</summary>
   <p align="center"><img width="60%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/67139c3e-2d21-4c6a-885b-ab65703add68" /></p>
   </details>

* **Denoising Autoregressive Representation Learning**<br>
*Yazhe Li, Jorg Bornschein, Ting Chen*<br>
ArXiv'2024 [[Paper](https://arxiv.org/abs/2403.05196)]
   <details close>
   <summary>DARL Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Westlake-AI/A2MIM/assets/44519745/76b1d81e-9ea6-4830-93f9-6be1ddba3432" /></p>
   </details>

<p align="right">(<a href="#top">back to top</a>)</p>

### MIM with Constrastive Learning

* **MST: Masked Self-Supervised Transformer for Visual Representation**<br>
*Zhaowen Li, Zhiyang Chen, Fan Yang, Wei Li, Yousong Zhu, Chaoyang Zhao, Rui Deng, Liwei Wu, Rui Zhao, Ming Tang, Jinqiao Wang*<br>
NeurIPS'2021 [[Paper](https://arxiv.org/abs/2106.05656)]
   <details close>
   <summary>MST Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204311330-9652d5d0-4b94-4f9a-afcd-efc12c712279.png" /></p>
   </details>

* **Are Large-scale Datasets Necessary for Self-Supervised Pre-training**<br>
*Alaaeldin El-Nouby, Gautier Izacard, Hugo Touvron, Ivan Laptev, Hervé Jegou, Edouard Grave*<br>
ArXiv'2021 [[Paper](https://arxiv.org/abs/2112.10740)]
   <details close>
   <summary>SplitMask Framework</summary>
   <p align="center"><img width="60%" src="https://user-images.githubusercontent.com/44519745/204311839-6f1310c9-88b2-4f43-90ff-927cf8aba720.png" /></p>
   </details>

* **Masked Siamese Networks for Label-Efficient Learning**<br>
*Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Michael Rabbat, Nicolas Ballas*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2204.07141)]
[[Code](https://github.com/facebookresearch/msn)]
   <details close>
   <summary>MSN Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204312102-a35d65ac-61e6-46ba-bb86-6c18b8562966.png" /></p>
   </details>

* **Siamese Image Modeling for Self-Supervised Vision Representation Learning**<br>
*Chenxin Tao, Xizhou Zhu, Gao Huang, Yu Qiao, Xiaogang Wang, Jifeng Dai*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2206.01204)]
[[Code](https://github.com/fundamentalvision/Siamese-Image-Modeling)]
   <details close>
   <summary>SIM Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204312408-fe573880-62ac-4f6e-b7ed-c9163f0cea96.png" /></p>
   </details>

* **Masked Contrastive Representation Learning**<br>
*Yuchong Yao, Nandakishor Desai, Marimuthu Palaniswami*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2211.06012)]
   <details close>
   <summary>MACRL Framework</summary>
   <p align="center"><img width="70%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/29af2ae4-5629-480b-93f7-2dd14a370890" /></p>
   </details>

* **Masked Image Modeling with Denoising Contrast**<br>
*Kun Yi, Yixiao Ge, Xiaotong Li, Shusheng Yang, Dian Li, Jianping Wu, Ying Shan, Xiaohu Qie*<br>
ICLR'2023 [[Paper](https://arxiv.org/abs/2205.09616)]
[[Code](https://github.com/TencentARC/ConMIM)]
   <details close>
   <summary>ConMIM Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204312585-13d5094b-c90c-4ab6-88d1-b88d46d8ae62.png" /></p>
   </details>

* **RePre: Improving Self-Supervised Vision Transformer with Reconstructive Pre-training**<br>
*Luya Wang, Feng Liang, Yangguang Li, Honggang Zhang, Wanli Ouyang, Jing Shao*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2201.06857)]
   <details close>
   <summary>RePre Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204312825-03953a52-0c1a-4f7e-bf12-e13841c2d371.png" /></p>
   </details>

* **Masked Siamese ConvNets**<br>
*Li Jing, Jiachen Zhu, Yann LeCun*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2206.07700)]
   <details close>
   <summary>MSCN Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/216648027-99790176-87fa-4fc6-ad5f-a8fe255c60e6.png" /></p>
   </details>

* **Contrastive Masked Autoencoders are Stronger Vision Learners**<br>
*Zhicheng Huang, Xiaojie Jin, Chengze Lu, Qibin Hou, Ming-Ming Cheng, Dongmei Fu, Xiaohui Shen, Jiashi Feng*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2207.13532)]
[[Code](https://github.com/ZhichengHuang/CMAE)]
   <details close>
   <summary>CMAE Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204313292-54630e16-e8ea-4281-a922-1b08c860e721.png" /></p>
   </details>

* **A simple, efficient and scalable contrastive masked autoencoder for learning visual representations**<br>
*Shlok Mishra, Joshua Robinson, Huiwen Chang, David Jacobs, Aaron Sarna, Aaron Maschinot, Dilip Krishnan*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2210.16870)]
   <details close>
   <summary>CAN Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204313772-7c0bf6d4-8df1-4b05-8733-da5024513e10.png" /></p>
   </details>

* **MimCo: Masked Image Modeling Pre-training with Contrastive Teacher**<br>
*Qiang Zhou, Chaohui Yu, Hao Luo, Zhibin Wang, Hao Li*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2209.03063)]
   <details close>
   <summary>MimCo Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/216651122-8fe6a039-37a8-4bec-8988-2760006da0af.png" /></p>
   </details>

* **Contextual Image Masking Modeling via Synergized Contrasting without View Augmentation for Faster and Better Visual Pretraining**<br>
*Shaofeng Zhang, Feng Zhu, Rui Zhao, Junchi Yan*<br>
ICLR'2023 [[Paper](https://openreview.net/forum?id=A3sgyt4HWp)]
[[Code](https://github.com/Sherrylone/ccMIM)]
   <details close>
   <summary>ccMIM Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204314041-63c5e06d-b870-482d-8f6b-e70e1af9d642.png" /></p>
   </details>

* **How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders**<br>
*Qi Zhang, Yifei Wang, Yisen Wang*<br>
NeurIPS'2022 [[Paper](https://arxiv.org/abs/2210.08344)]
[[Code](https://github.com/zhangq327/U-MAE)]
   <details close>
   <summary>U-MAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/234359652-b34cb444-1c6b-4721-94e3-6bd60347ca55.png" /></p>
   </details>

* **Layer Grafted Pre-training: Bridging Contrastive Learning And Masked Image Modeling For Label-Efficient Representations**<br>
*Ziyu Jiang, Yinpeng Chen, Mengchen Liu, Dongdong Chen, Xiyang Dai, Lu Yuan, Zicheng Liu, Zhangyang Wang*<br>
ICLR'2023 [[Paper](https://openreview.net/forum?id=jwdqNwyREyh)]
[[Code](https://github.com/VITA-Group/layerGraftedPretraining_ICLR23)]
   <details close>
   <summary>Layer Grafted Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/224830983-13cfcbf5-f1df-481b-9e7c-24667d041fe4.png" /></p>
   </details>

* **DropPos: Pre-Training Vision Transformers by Reconstructing Dropped Positions**<br>
*Haochen Wang, Junsong Fan, Yuxi Wang, Kaiyou Song, Tong Wang, Zhaoxiang Zhang*<br>
NeurIPS'2023 [[Paper](https://arxiv.org/abs/2309.03576)]
[[Code](https://github.com/Haochen-Wang409/DropPos)]
   <details close>
   <summary>DropPos Framework</summary>
   <p align="center"><img width="80%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/275285480-27b8a822-f298-4887-b012-249948c29a22.png" /></p>
   </details>

* **Rejuvenating image-GPT as Strong Visual Representation Learners**<br>
*Sucheng Ren, Zeyu Wang, Hongru Zhu, Junfei Xiao, Alan Yuille, Cihang Xie*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2312.02147)]
[[Code](https://github.com/OliverRensu/D-iGPT)]
   <details close>
   <summary>D-iGPT Framework</summary>
   <p align="center"><img width="60%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/13002c8d-179e-484d-846c-9bd4b25b58d8" /></p>
   </details>

* **CoMAE: Single Model Hybrid Pre-training on Small-Scale RGB-D Datasets**<br>
*Jiange Yang, Sheng Guo, Gangshan Wu, Limin Wang*<br>
AAAI'2023 [[Paper](https://arxiv.org/abs/2302.06148)]
[[Code](https://github.com/MCG-NJU/CoMAE)]
   <details close>
   <summary>CoMAE Framework</summary>
   <p align="center"><img width="80%" src="https://github.com/MCG-NJU/CoMAE/blob/main/framework.png" /></p>
   </details>

<p align="right">(<a href="#top">back to top</a>)</p>

### MIM for Transformers and CNNs

* **Context Encoders: Feature Learning by Inpainting**<br>
*Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros*<br>
CVPR'2016 [[Paper](https://arxiv.org/abs/1604.07379)]
[[Code](https://github.com/pathak22/context-encoder)]
   <details close>
   <summary>Context-Encoder Framework</summary>
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/204314544-4ad0e4a8-f7b8-47f9-80e9-67ef87c1b14a.png" /></p>
   </details>

* **Corrupted Image Modeling for Self-Supervised Visual Pre-Training**<br>
*Yuxin Fang, Li Dong, Hangbo Bao, Xinggang Wang, Furu Wei*<br>
ICLR'2023 [[Paper](https://arxiv.org/abs/2202.03382)]
   <details close>
   <summary>CIM Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204315003-182e9ba5-5ab3-4d84-9544-8e0a3d8590c5.png" /></p>
   </details>

* **Architecture-Agnostic Masked Image Modeling - From ViT back to CNN**<br>
*Siyuan Li, Di Wu, Fang Wu, Zelin Zang, Stan.Z.Li*<br>
ICML'2023 [[Paper](https://arxiv.org/abs/2205.13943)]
[[Code](https://github.com/Westlake-AI/openmixup)] [[project](https://github.com/Westlake-AI/A2MIM)]
   <details close>
   <summary>A2MIM Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204314681-d953cffc-8ba7-481c-925e-c89084f83c56.png" /></p>
   </details>

* **Masked Frequency Modeling for Self-Supervised Visual Pre-Training**<br>
*Jiahao Xie, Wei Li, Xiaohang Zhan, Ziwei Liu, Yew Soon Ong, Chen Change Loy*<br>
ICLR'2023 [[Paper](https://arxiv.org/abs/2206.07706)]
[[Code](https://github.com/CoinCheung/MFM)]
   <details close>
   <summary>MFM Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204315329-1a58598f-35cb-439c-91ee-303ddd36fa6c.png" /></p>
   </details>

* **MixMAE: Mixed and Masked Autoencoder for Efficient Pretraining of Hierarchical Vision Transformers**<br>
*Jihao Liu, Xin Huang, Jinliang Zheng, Yu Liu, Hongsheng Li*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2205.13137)]
[[Code](https://github.com/Sense-X/MixMIM)]
   <details close>
   <summary>MixMAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204315480-5c59ed60-7b5f-4da9-85fb-551a961fd731.png" /></p>
   </details>

* **Masked Autoencoders are Robust Data Augmentors**<br>
*Haohang Xu, Shuangrui Ding, Xiaopeng Zhang, Hongkai Xiong, Qi Tian*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2206.04846)]
[[Code](https://github.com/haohang96/mra)]
   <details close>
   <summary>MRA Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204315634-212c14b9-7d6d-4ad0-880b-35cafb623249.png" /></p>
   </details>

* **Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling**<br>
*Keyu Tian, Yi Jiang, Qishuai Diao, Chen Lin, Liwei Wang, Zehuan Yuan*<br>
ICLR'2023 [[Paper](https://arxiv.org/abs/2301.03580)]
[[Code](https://github.com/keyu-tian/spark)]
   <details close>
   <summary>SparK Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204315983-d5a24e55-fab4-4336-a1ed-3428a997aebd.png" /></p>
   </details>

* **ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders**<br>
*Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2301.00808)]
[[Code](https://github.com/facebookresearch/ConvNeXt-V2)]
   <details close>
   <summary>ConvNeXt.V2 Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/211898674-caa94c81-9aea-4544-8f5f-2cf410724bb4.png" /></p>
   </details>

* **RevColV2: Exploring Disentangled Representations in Masked Image Modeling**<br>
*Qi Han, Yuxuan Cai, Xiangyu Zhang*<br>
NeurIPS'2023 [[Paper](https://arxiv.org/abs/2309.01005)]
[[Code](https://github.com/megvii-research/RevCol)]
   <details close>
   <summary>RevCol.V2 Framework</summary>
   <p align="center"><img width="85%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/275285082-7c367ac3-bfce-4502-bf77-b7402b2f04e4.png" /></p>
   </details>

* **Masked Capsule Autoencoders**<br>
*Miles Everett, Mingjun Zhong, Georgios Leontidis*<br>
arXiv'2024 [[Paper](https://arxiv.org/abs/2403.04724)]
   <details close>
   <summary>MCAE Framework</summary>
   <p align="center"><img width="80%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/1708e5a1-6e4e-4d74-926d-b474c3069841" /></p>
   </details>

* **MixMask: Revisiting Masking Strategy for Siamese ConvNets**<br>
*Kirill Vishniakov, Eric Xing, Zhiqiang Shen*<br>
BMVC'2024 [[Paper](https://arxiv.org/abs/2210.11456)]
[[Code](https://github.com/LightnessOfBeing/MixMask)]
   <details close>
   <summary>MixMask Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/user-attachments/assets/e997db8d-cebe-449a-8fdb-14682955147f" /></p>
   </details>

<p align="right">(<a href="#top">back to top</a>)</p>

### MIM with Advanced Masking

* **MST: Masked Self-Supervised Transformer for Visual Representation**<br>
*Zhaowen Li, Zhiyang Chen, Fan Yang, Wei Li, Yousong Zhu, Chaoyang Zhao, Rui Deng, Liwei Wu, Rui Zhao, Ming Tang, Jinqiao Wang*<br>
NeurIPS'2021 [[Paper](https://arxiv.org/abs/2106.05656)]
   <details close>
   <summary>MST Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204311330-9652d5d0-4b94-4f9a-afcd-efc12c712279.png" /></p>
   </details>

* **Adversarial Masking for Self-Supervised Learning**<br>
*Yuge Shi, N. Siddharth, Philip H.S. Torr, Adam R. Kosiorek*<br>
ICML'2022 [[Paper](https://arxiv.org/abs/2201.13100)]
[[Code](https://github.com/YugeTen/adios)]
   <details close>
   <summary>ADIOS Framework</summary>
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/204316447-66b223b7-1518-477d-9d5f-66bd3148eecd.png" /></p>
   </details>

* **What to Hide from Your Students: Attention-Guided Masked Image Modeling**<br>
*Ioannis Kakogeorgiou, Spyros Gidaris, Bill Psomas, Yannis Avrithis, Andrei Bursuc, Konstantinos Karantzalos, Nikos Komodakis*<br>
ECCV'2022 [[Paper](https://arxiv.org/abs/2203.12719)]
[[Code](https://github.com/gkakogeorgiou/attmask)]
   <details close>
   <summary>AttMask Framework</summary>
   <p align="center"><img width="65%" src="https://user-images.githubusercontent.com/44519745/204316717-191ef56d-c703-4b12-9c71-28bd14371d32.png" /></p>
   </details>

* **Uniform Masking: Enabling MAE Pre-training for Pyramid-based Vision Transformers with Locality**<br>
*Xiang Li, Wenhai Wang, Lingfeng Yang, Jian Yang*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2205.10063)]
[[Code](https://github.com/implus/um-mae)]
   <details close>
   <summary>UnMAE Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204316895-a04d2141-4dc9-47db-9176-001d71dcc704.png" /></p>
   </details>

* **SemMAE: Semantic-Guided Masking for Learning Masked Autoencoders**<br>
*Gang Li, Heliang Zheng, Daqing Liu, Chaoyue Wang, Bing Su, Changwen Zheng*<br>
NeurIPS'2022 [[Paper](https://arxiv.org/abs/2206.10207)]
[[Code](https://github.com/ucasligang/semmae)]
   <details close>
   <summary>SemMAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204317096-f6ade707-6f66-4826-823e-e14d0784b960.png" /></p>
   </details>

* **Good helper is around you: Attention-driven Masked Image Modeling**<br>
*Zhengqi Liu, Jie Gui, Hao Luo*<br>
AAAI'2023 [[Paper](https://arxiv.org/abs/2211.15362)]
[[Code](https://github.com/guijiejie/AMT)]
   <details close>
   <summary>AMT Framework</summary>
   <p align="center"><img width="80%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/2c596060-ab99-4ad8-a879-819a7cfb986b" /></p>
   </details>

* **Hard Patches Mining for Masked Image Modeling**<br>
*Haochen Wang, Kaiyou Song, Junsong Fan, Yuxi Wang, Jin Xie, Zhaoxiang Zhang*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2304.05919)]
[[Code](https://github.com/Haochen-Wang409/HPM)]
   <details close>
   <summary>HPM Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/232319362-d6a5419f-6a95-4405-a615-f8ded42c1896.png" /></p>
   </details>

* **AdaMAE: Adaptive Masking for Efficient Spatiotemporal Learning with Masked Autoencoders**<br>
*Wele Gedara Chaminda Bandara, Naman Patel, Ali Gholami, Mehdi Nikkhah, Motilal Agrawal, Vishal M. Patel*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2211.09120)]
[[Code](https://github.com/wgcban/adamae)]
   <details close>
   <summary>AdaMAE Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Westlake-AI/Awesome-Mixup/assets/44519745/9a47b15b-00b1-45c5-a06d-cce1ce29d9b3" /></p>
   </details>

* **Improving Masked Autoencoders by Learning Where to Mask**<br>
*Haijian Chen, Wendong Zhang, Yunbo Wang, Xiaokang Yang*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2303.06583)]
   <details close>
   <summary>AutoMAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/236310631-a11eee44-3e70-414f-9caa-ad09a231ce31.png" /></p>
   </details>

* **Learning with Noisy labels via Self-supervised Adversarial Noisy Masking**<br>
*Yuanpeng Tu, Boshen Zhang, Yuxi Li, Liang Liu, Jian Li, Jiangning Zhang, Yabiao Wang, Chengjie Wang, Cai Rong Zhao*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2302.06805)]
[[Code](https://github.com/yuanpengtu/SANM)]

<p align="right">(<a href="#top">back to top</a>)</p>

### MIM for Multi-Modality

* **VL-BERT: Pre-training of Generic Visual-Linguistic Representations**<br>
*Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, Jifeng Dai*<br>
ICLR'2020 [[Paper](https://arxiv.org/abs/1908.08530)]
[[Code](https://github.com/jackroos/VL-BERT)]
   <details close>
   <summary>VL-BERT Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/278494134-b57d357e-1b3e-44ac-9de5-1303e4b7dbcf.png" /></p>
   </details>

* **MaskCLIP: Masked Self-Distillation Advances Contrastive Language-Image Pretraining**<br>
*Xiaoyi Dong, Jianmin Bao, Yinglin Zheng, Ting Zhang, Dongdong Chen, Hao Yang, Ming Zeng, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2208.12262)]
[[Code](https://github.com/LightDXY/MaskCLIP)]
   <details close>
   <summary>MaskCLIP Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/209704378-5c467c07-8096-441d-b8a4-37fe27d1ac07.png" /></p>
   </details>

* **Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks**<br>
*Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, Aniruddha Kembhavi*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2206.08916)]
[[Code](https://arxiv.org/abs/2206.08916)]
   <details close>
   <summary>Unified-IO Framework</summary>
   <p align="center"><img width="85%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/278491185-0babafa8-c7d5-4976-94d5-f02a12d605dd.png" /></p>
   </details>

* **Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks**<br>
*Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, Furu Wei*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2208.10442)]
[[Code](https://github.com/microsoft/unilm/tree/master/beit)]
   <details close>
   <summary>BEiT.V3 Framework</summary>
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/204308318-b1d80584-2b7e-4c35-bb68-171c9bfaf299.png" /></p>
   </details>

* **Masked Vision and Language Modeling for Multi-modal Representation Learning**<br>
*Gukyeong Kwon, Zhaowei Cai, Avinash Ravichandran, Erhan Bas, Rahul Bhotika, Stefano Soatto*<br>
ICLR'2023 [[Paper](https://arxiv.org/abs/2208.02131)]
   <details close>
   <summary>MaskVLM Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204310845-3e7777dc-5726-4c94-9506-8f88efd1966b.png" /></p>
   </details>

* **Scaling Language-Image Pre-training via Masking**<br>
*Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, Kaiming He*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2212.00794)]
   <details close>
   <summary>FLIP Framework</summary>
   <p align="center"><img width="55%" src="https://user-images.githubusercontent.com/44519745/209705278-368b8125-cb14-4800-8523-09800b1728d4.png" /></p>
   </details>

* **All in Tokens: Unifying Output Space of Visual Tasks via Soft Token**<br>
*Jia Ning, Chen Li, Zheng Zhang, Zigang Geng, Qi Dai, Kun He, Han Hu*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2301.02229)]
   <details close>
   <summary>AiT Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/18166057-152e-4547-abd8-747df7d5ffb4" /></p>
   </details>

* **Attentive Mask CLIP**<br>
*Yifan Yang, Weiquan Huang, Yixuan Wei, Houwen Peng, Xinyang Jiang, Huiqiang Jiang, Fangyun Wei, Yin Wang, Han Hu, Lili Qiu, Yuqing Yang*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2212.08653)]
   <details close>
   <summary>A-CLIP Framework</summary>
   <p align="center"><img width="55%" src="https://user-images.githubusercontent.com/44519745/209704869-a0eee2b2-0b21-4be1-8bfa-074758f3b4a2.png" /></p>
   </details>

* **MultiModal-GPT: A Vision and Language Model for Dialogue with Humans**<br>
*Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, Kai Chen*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2305.04790)]
[[Code](https://github.com/open-mmlab/Multimodal-GPT)]
   <details close>
   <summary>MultiModal-GPT Framework</summary>
   <p align="center"><img width="75%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/a492f540-e287-4c2a-8879-7812cd2b2767" /></p>
   </details>

* **VL-GPT: A Generative Pre-trained Transformer for Vision and Language Understanding and Generation**<br>
*Jinguo Zhu, Xiaohan Ding, Yixiao Ge, Yuying Ge, Sijie Zhao, Hengshuang Zhao, Xiaohua Wang, Ying Shan*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2312.09251)]
[[Code](https://github.com/ailab-cvc/vl-gpt)]
   <details close>
   <summary>VL-GPT Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Westlake-AI/MogaNet/assets/44519745/5b7cfab1-5945-4080-afe1-46299cb82e72" /></p>
   </details>

* **Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action**<br>
*Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, Aniruddha Kembhavi*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2312.17172)]
[[Code](https://github.com/allenai/unified-io-2)]
   <details close>
   <summary>Unified-IO 2 Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/b4ad7a11-ac60-40bb-990f-6ab4586035b2" /></p>
   </details>

* **Self-Guided Masked Autoencoders for Domain-Agnostic Self-Supervised Learning**<br>
*Johnathan Wenjia Xie, Yoonho Lee, Annie S Chen, Chelsea Finn*<br>
ICLR'2024 [[Paper](https://openreview.net/forum?id=HiYMiZYwkw)]
[[Code](https://github.com/Johnathan-Xie/sma)]
   <details close>
   <summary>SMA Framework</summary>
   <p align="center"><img width="80%" src="https://github.com/user-attachments/assets/11292f56-a8e2-4ea5-b9a1-3fe34d29b261" /></p>
   </details>

### MIM for Vision Generalist Model

* **A Generalist Agent**<br>
*Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, Tom Eccles, Jake Bruce, Ali Razavi, Ashley Edwards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol Vinyals, Mahyar Bordbar, Nando de Freitas*<br>
TMLR'2022 [[Paper](https://arxiv.org/abs/2205.06175)]
[[Code](https://github.com/OrigamiDream/gato)]
   <details close>
   <summary>Gato Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/user-attachments/assets/439489b1-90c4-4004-87fa-4802e4962162" /></p>
   </details>

* **Scaling Autoregressive Models for Content-Rich Text-to-Image Generation**<br>
*Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, Yonghui Wu*<br>
TMLR'2022 [[Paper](https://openreview.net/forum?id=AFDcYJKhND)]
   <details close>
   <summary>Parti Framework</summary>
   <p align="center"><img width="80%" src="https://github.com/user-attachments/assets/4b00770f-998d-4be4-8bc7-3a3c8bf2e177" /></p>
   </details>

* **Images Speak in Images: A Generalist Painter for In-Context Visual Learning**<br>
*Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, Tiejun Huang*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2212.02499)]
[[Code](https://github.com/baaivision/Painter)]
   <details close>
   <summary>Painter Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/de7240fd-f04c-411f-9ae1-f2e4839236ee" /></p>
   </details>

* **InstructCV: Instruction-Tuned Text-to-Image Diffusion Models as Vision Generalists**<br>
*Yulu Gan, Sungwoo Park, Alexander Schubert, Anthony Philippakis, Ahmed M. Alaa*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2310.00390)]
[[Code](https://github.com/AlaaLab/InstructCV)]
   <details close>
   <summary>InstructCV Framework</summary>
   <p align="center"><img width="95%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/7a875d40-48fc-4570-9547-1059e6a742b4" /></p>
   </details>

* **InstructDiffusion: A Generalist Modeling Interface for Vision Tasks**<br>
*Zigang Geng, Binxin Yang, Tiankai Hang, Chen Li, Shuyang Gu, Ting Zhang, Jianmin Bao, Zheng Zhang, Han Hu, Dong Chen, Baining Guo*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2309.03895)]
   <details close>
   <summary>InstructDiffusion Framework</summary>
   <p align="center"><img width="95%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/925312fe-b020-4091-b5fa-b1e777e06309" /></p>
   </details>

* **Sequential Modeling Enables Scalable Learning for Large Vision Models**<br>
*Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan Yuille, Trevor Darrell, Jitendra Malik, Alexei A Efros*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2312.00785)]
[[Code](https://yutongbai.com/lvm.html)]
   <details close>
   <summary>LVM Framework</summary>
   <p align="center"><img width="75%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/c5bfa7df-7f7c-49bf-9674-749b1b5ebd8e" /></p>
   </details>

<p align="right">(<a href="#top">back to top</a>)</p>

### Image Generation

* **Discrete Variational Autoencoders**<br>
*Jason Tyler Rolfe*<br>
ICLR'2017
[[Paper](https://arxiv.org/abs/1609.02200)]
[[Code](https://github.com/openai/DALL-E)]

* **Neural Discrete Representation Learning**<br>
*Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu*<br>
NeurIPS'2017
[[Paper](https://proceedings.neurips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)]
[[Code](https://github.com/ritheshkumar95/pytorch-vqvae)]

* **Theory and Experiments on Vector Quantized Autoencoders (EM VQ-VAE)**<br>
*Aurko Roy, Ashish Vaswani, Arvind Neelakantan, Niki Parmar*<br>
Arxiv'2018
[[Paper](https://arxiv.org/abs/1805.11063)]
[[Code](https://github.com/jaywalnut310/Vector-Quantized-Autoencoders)]

* **DVAE: Discrete Variational Autoencoders with Relaxed Boltzmann Priors**<br>
*Arash Vahdat, Evgeny Andriyash, William G. Macready*<br>
NeurIPS'2018
[[Paper](https://arxiv.org/abs/1805.07445)]
[[Code](https://github.com/xmax1/dvae)]

* **DVAE++: Discrete Variational Autoencoders with Overlapping Transformations**<br>
*Arash Vahdat, William G. Macready, Zhengbing Bian, Amir Khoshaman, Evgeny Andriyash*<br>
ICML'2018
[[Paper](https://arxiv.org/abs/1802.04920)]
[[Code](https://github.com/xmax1/dvae)]

* **Generating Diverse High-Fidelity Images with VQ-VAE-2**<br>
*Ali Razavi, Aaron van den Oord, Oriol Vinyals*<br>
NeurIPS'2019
[[Paper](https://proceedings.neurips.cc/paper/2019/file/5f8e2fa1718d1bbcadf1cd9c7a54fb8c-Paper.pdf)]
[[Code](https://github.com/rosinality/vq-vae-2-pytorch)]

* **Generative Pretraining from Pixels**<br>
*Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, David Luan, Ilya Sutskever*<br>
ICML'2020 [[Paper](http://proceedings.mlr.press/v119/chen20s/chen20s.pdf)]
[[Code](https://github.com/openai/image-gpt)]
   <details close>
   <summary>iGPT Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204300433-a0b6b25b-9f6f-431b-bbfd-19169d8cbca6.png" /></p>
   </details>

* **Taming Transformers for High-Resolution Image Synthesis**<br>
*Patrick Esser, Robin Rombach, Björn Ommer*<br>
CVPR'2021 [[Paper](https://arxiv.org/abs/2012.09841)]
[[Code](https://github.com/CompVis/taming-transformers)]
   <details close>
   <summary>VQGAN Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/260847328-ef6d17b6-72a2-4b85-a89a-d48dba273c1e.png" /></p>
   </details>

* **MaskGIT: Masked Generative Image Transformer**<br>
*Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman*<br>
CVPR'2022 [[Paper](https://arxiv.org/abs/2202.04200)]
[[Code](https://github.com/google-research/maskgit)]
   <details close>
   <summary>MaskGIT Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/261109349-c824eba4-edc5-4d00-8480-1a3d2929dcc6.png" /></p>
   </details>

* **ERNIE-ViLG: Unified Generative Pre-training for Bidirectional Vision-Language Generation**<br>
*Han Zhang, Weichong Yin, Yewei Fang, Lanxin Li, Boqiang Duan, Zhihua Wu, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang*<br>
Arxiv'2021
[[Paper](https://arxiv.org/abs/2112.15283)] 
[[Project](https://wenxin.baidu.com/wenxin/ernie-vilg)]

* **NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion**<br>
*Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, Nan Duan*<br>
Arxiv'2021
[[Paper](https://arxiv.org/abs/2111.12417)] 
[[Code](https://github.com/microsoft/NUWA)] 

* **ImageBART: Bidirectional Context with Multinomial Diffusion for Autoregressive Image Synthesis**<br>
*Patrick Esser, Robin Rombach, Andreas Blattmann, Björn Ommer*<br>
NeurIPS'2021
[[Paper](https://openreview.net/pdf?id=-1AAgrS5FF)] 
[[Code](https://github.com/CompVis/imagebart)] 
[[Project](https://compvis.github.io/imagebart/)]

* **Vector-quantized Image Modeling with Improved VQGAN**<br>
*Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, Yonghui Wu*<br>
ICLR'2022 [[Paper](https://arxiv.org/abs/2110.04627)]
[[Code](https://github.com/lucidrains/DALLE2-pytorch)]
   <details close>
   <summary>ViT-VQGAN Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/260846200-0a7fd6e6-170a-435d-9f7b-a020bde64bc5.png" /></p>
   </details>

* **Self-supervision through Random Segments with Autoregressive Coding (RandSAC)**<br>
*Tianyu Hua, Yonglong Tian, Sucheng Ren, Michalis Raptis, Hang Zhao, Leonid Sigal*<br>
ICLR'2023 [[Paper](https://arxiv.org/abs/2203.12054)]
   <details close>
   <summary>RandSAC Framework</summary>
   <p align="center"><img width="80%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/261103618-f2aa7486-a09f-4f50-a84d-fb367c621d04.png" /></p>
   </details>

* **MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis**<br>
*Tianhong Li, Huiwen Chang, Shlok Kumar Mishra, Han Zhang, Dina Katabi, Dilip Krishnan*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2211.09117)]
[[Code](https://github.com/lth14/mage)]
   <details close>
   <summary>MAGE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/224833197-6d95863c-cb83-4d9d-a900-b4f61baba785.png" /></p>
   </details>

* **Not All Image Regions Matter: Masked Vector Quantization for Autoregressive Image Generation**<br>
*Mengqi Huang, Zhendong Mao, Quan Wang, Yongdong Zhang*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2305.13607)]
[[Code](https://github.com/CrossmodalGroup/MaskedVectorQuantization)]
   <details close>
   <summary>MQ-VAE Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/bf859af5-5fbb-46f4-a65d-ca8fba896981" /></p>
   </details>

* **Towards Accurate Image Coding: Improved Autoregressive Image Generation with Dynamic Vector Quantization**<br>
*Mengqi Huang, Zhendong Mao, Zhuowei Chen, Yongdong Zhang*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2305.11718)]
[[Code](https://github.com/CrossmodalGroup/DynamicVectorQuantization)]
   <details close>
   <summary>DQ-VAE Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/72f8eb15-97f2-4c8c-9597-dfdb1edc44d2" /></p>
   </details>

* **Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment**<br>
*Hao Liu, Wilson Yan, Pieter Abbeel*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2302.00902)]
[[Code](https://github.com/lhao499/language-quantized-autoencoders)]
   <details close>
   <summary>LQAE Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/260847938-cab2ff4f-e504-4ac1-a7bf-d11d710bb74a.png" /></p>
   </details>

* **SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs**<br>
*Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David A. Ross, Irfan Essa, Yonatan Bisk, Ming-Hsuan Yang, Kevin Murphy, Alexander G. Hauptmann, Lu Jiang*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2306.17842)]
[[Code](https://github.com/google-research/magvit/)]
   <details close>
   <summary>SPAE Framework</summary>
   <p align="center"><img width="85%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/258768117-d71a41d6-69d7-484f-868f-9f38b08c936c.png" /></p>
   </details>

* **Text-Conditioned Sampling Framework for Text-to-Image Generation with Masked Generative Models**<br>
*Jaewoong Lee, Sangwon Jang, Jaehyeong Jo, Jaehong Yoon, Yunji Kim, Jin-Hwa Kim, Jung-Woo Ha, Sung Ju Hwang*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2304.01515)]
   <details close>
   <summary>TCTS Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204300433-a0b6b25b-9f6f-431b-bbfd-19169d8cbca6.png" /></p>
   </details>

* **Diffusion Models as Masked Autoencoders**<br>
*Chen Wei, Karttikeya Mangalam, Po-Yao Huang, Yanghao Li, Haoqi Fan, Hu Xu, Huiyu Wang, Cihang Xie, Alan Yuille, Christoph Feichtenhofer*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2304.03283)]
[[Code](https://weichen582.github.io/diffmae.html)]
   <details close>
   <summary>TCTS Framework</summary>
   <p align="center"><img width="55%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273509400-caf78224-79a2-4ccc-90af-a7dc5b8518a4.png" /></p>
   </details>

* **Masked Diffusion Transformer is a Strong Image Synthesizer**<br>
*Shanghua Gao, Pan Zhou, Ming-Ming Cheng, Shuicheng Yan*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2303.14389)]
[[Code](https://github.com/sail-sg/MDT)]
   <details close>
   <summary>MDT Framework</summary>
   <p align="center"><img width="60%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273515639-af09f81b-c1d7-4d39-aa1a-6ffde172c030.png" /></p>
   </details>

* **Self-conditioned Image Generation via Generating Representations**<br>
*Tianhong Li, Dina Katabi, Kaiming He*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2312.03701)]
[[Code](https://github.com/LTH14/rcg)]
   <details close>
   <summary>RCG Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/88f210d7-2cba-456d-8ab2-3a807c153ea3" /></p>
   </details>

* **OneLLM: One Framework to Align All Modalities with Language**<br>
*Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua Lin, Yu Qiao, Peng Gao, Xiangyu Yue*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2312.03700)]
[[Code](https://github.com/csuhan/OneLLM)]
   <details close>
   <summary>OneLLM Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/88f210d7-2cba-456d-8ab2-3a807c153ea3" /></p>
   </details>

* **MDTv2: Masked Diffusion Transformer is a Strong Image Synthesizer**<br>
*Shanghua Gao, Pan Zhou, Ming-Ming Cheng, Shuicheng Yan*<br>
ArXiv'2024 [[Paper](https://arxiv.org/abs/2303.14389)]
[[Code](https://github.com/sail-sg/MDT)]
   <details close>
   <summary>MDTv2 Framework</summary>
   <p align="center"><img width="65%" src="https://github.com/Westlake-AI/openmixup/assets/44519745/42fc03e2-a517-4ff2-a008-49a5a1b2b1c1" /></p>
   </details>

* **Beyond Text: Frozen Large Language Models in Visual Signal Comprehension**<br>
*Lei Zhu, Fangyun Wei, Yanye Lu*<br>
CVPR'2024 [[Paper](https://arxiv.org/abs/2403.07874)]
[[Code](https://github.com/zh460045050/V2L-Tokenizer)]
   <details close>
   <summary>V2L Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/9a645669-5202-491a-a25c-432de5d2ca32" /></p>
   </details>

* **Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation**<br>
*Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, Zehuan Yuan*<br>
ArXiv'2024 [[Paper](https://arxiv.org/abs/2406.06525)]
[[Code](https://github.com/FoundationVision/LlamaGen)]

* **Scaling the Codebook Size of VQGAN to 100,000 with a Utilization Rate of 99%**<br>
*Lei Zhu, Fangyun Wei, Yanye Lu, Dong Chen*<br>
ArXiv'2024 [[Paper](https://arxiv.org/abs/2406.11837)]
[[Code](https://github.com/zh460045050/VQGAN-LC)]
   <details close>
   <summary>VQGAN-LC Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Westlake-AI/openmixup/assets/44519745/407b6e71-dcb8-454b-aba8-9bdcca1eb0df" /></p>
   </details>

<p align="right">(<a href="#top">back to top</a>)</p>

## MIM for CV Downstream Tasks

### Object Detection and Segmentation

* **Unleashing Vanilla Vision Transformer with Masked Image Modeling for Object Detection**<br>
*Yuxin Fang, Shusheng Yang, Shijie Wang, Yixiao Ge, Ying Shan, Xinggang Wang*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2204.02964)]
[[Code](https://github.com/hustvl/MIMDet)]
   <details close>
   <summary>MIMDet Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/207723589-e1fd22e3-0719-422e-b94d-371c51b164e5.png" /></p>
   </details>

* **SeqCo-DETR: Sequence Consistency Training for Self-Supervised Object Detection with Transformers**<br>
*Guoqiang Jin, Fan Yang, Mingshan Sun, Ruyi Zhao, Yakun Liu, Wei Li, Tianpeng Bao, Liwei Wu, Xingyu Zeng, Rui Zhao*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2303.08481)]
   <details close>
   <summary>SeqCo-DETR Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/226742022-7b823b2b-a08c-4579-8f33-8f35b282069a.png" /></p>
   </details>

* **Integrally Pre-Trained Transformer Pyramid Networks**<br>
*Yunjie Tian, Lingxi Xie, Zhaozhi Wang, Longhui Wei, Xiaopeng Zhang, Jianbin Jiao, Yaowei Wang, Qi Tian, Qixiang Ye*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2211.12735)]
[[Code](https://github.com/sunsmarterjie/iTPN)]
   <details close>
   <summary>iTPN Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/231279943-851af288-fe43-44ba-aa85-3ca6ee72a247.png" /></p>
   </details>

* **PiMAE: Point Cloud and Image Interactive Masked Autoencoders for 3D Object Detection**<br>
*Anthony Chen, Kevin Zhang, Renrui Zhang, Zihan Wang, Yuheng Lu, Yandong Guo, Shanghang Zhang*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2303.08129)]
[[Code](https://github.com/BLVLab/PiMAE)]
   <details close>
   <summary>PiMAE Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/251236178-0e2a8530-d644-4c04-8a83-63c16c3743bb.png" /></p>
   </details>

* **Integrally Migrating Pre-trained Transformer Encoder-decoders for Visual Object Detection**<br>
*Yuan Liu, Songyang Zhang, Jiacheng Chen, Zhaohui Yu, Kai Chen, Dahua Lin*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2205.09613)]
[[Code](https://github.com/LiewFeng/imTED)]
   <details close>
   <summary>imTED Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/263503486-6daa9793-904f-46d7-882e-9a53606cdcd1.png" /></p>
   </details>

* **Masked Retraining Teacher-student Framework for Domain Adaptive Object Detection**<br>
*Zijing Zhao, Sitong Wei, Qingchao Chen, Dehui Li, Yifan Yang, Yuxin Peng, Yang Liu*<br>
ICCV'2023 [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Masked_Retraining_Teacher-Student_Framework_for_Domain_Adaptive_Object_Detection_ICCV_2023_paper.pdf)]
[[Code](https://github.com/JeremyZhao1998/MRT-release)]
   <details close>
   <summary>MRT Framework</summary>
   <p align="center"><img width="85%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273500384-73df36d2-3dbb-4883-a6f5-8a22edac7f58.png" /></p>
   </details>

* **Object Recognition as Next Token Prediction**<br>
*Kaiyu Yue, Bor-Chun Chen, Jonas Geiping, Hengduo Li, Tom Goldstein, Ser-Nam Lim*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2312.02142)]
[[Code](https://github.com/kaiyuyue/nxtp)]
   <details close>
   <summary>imTED Framework</summary>
   <p align="center"><img width="75%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/7badecf5-3714-4a48-bbf9-26ea91167626" /></p>
   </details>

* **EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything**<br>
*Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, Raghuraman Krishnamoorthi, Vikas Chandra*<br>
CVPR'2024 [[Paper](https://arxiv.org/abs/2312.00863)]
[[Code](https://github.com/yformer/EfficientSAM)]
   <details close>
   <summary>EfficientSAM Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/0369174b-b17d-432e-ad06-470b49a51a75" /></p>
   </details>

### Video Rrepresentation

* **VideoGPT: Video Generation using VQ-VAE and Transformers**<br>
*Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas*<br>
arXiv'2021 [[Paper](https://arxiv.org/abs/2104.10157)]
[[Code](https://github.com/wilson1yan/VideoGPT)]
   <details close>
   <summary>VideoGPT Framework</summary>
   <p align="center"><img width="70%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/260848752-da40d950-5154-4ba1-8702-9dcd448a83bc.png" /></p>
   </details>

* **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training**<br>
*Zhan Tong, Yibing Song, Jue Wang, Limin Wang*<br>
NeurIPS'2022 [[Paper](https://arxiv.org/abs/2203.12602)]
[[Code](https://github.com/MCG-NJU/VideoMAE)]
   <details close>
   <summary>VideoMAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/207724710-e4093d2e-8d6c-40b9-bf7d-ab519eb97dd2.png" /></p>
   </details>

* **Masked Autoencoders As Spatiotemporal Learners**<br>
*Christoph Feichtenhofer, Haoqi Fan, Yanghao Li, Kaiming He*<br>
NeurIPS'2022 [[Paper](https://arxiv.org/abs/2205.09113)]
[[Code](https://github.com/facebookresearch/SlowFast)]
   <details close>
   <summary>MAE Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/207725088-8bccb8df-a9c8-4ba6-b7cd-5f259c0959c1.png" /></p>
   </details>

* **Less is More: Consistent Video Depth Estimation with Masked Frames Modeling**<br>
*Yiran Wang, Zhiyu Pan, Xingyi Li, Zhiguo Cao, Ke Xian, Jianming Zhang*<br>
ACMMM'2022 [[Paper](https://arxiv.org/abs/2208.00380)]
[[Code](https://github.com/RaymondWang987/FMNet)]
   <details close>
   <summary>FMNet Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/207725289-efef0f35-77a7-4ad4-896a-f1c84503cbb5.png" /></p>
   </details>

* **MaskViT: Masked Visual Pre-Training for Video Prediction**<br>
*Agrim Gupta, Stephen Tian, Yunzhi Zhang, Jiajun Wu, Roberto Martín-Martín, Li Fei-Fei*<br>
CVPR'2022 [[Paper](https://arxiv.org/abs/2206.11894)]
[[Code](https://github.com/agrimgupta92/maskvit)]
   <details close>
   <summary>MaskViT Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/207725785-e8b125ff-22c1-451a-b1d8-101c13113189.png" /></p>
   </details>

* **BEVT: BERT Pretraining of Video Transformers**<br>
*Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Yu-Gang Jiang, Luowei Zhou, Lu Yuan*<br>
CVPR'2022 [[Paper](https://arxiv.org/abs/2112.01529)]
[[Code](https://github.com/xyzforever/BEVT)]
   <details close>
   <summary>BEVT Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/2b878f93-d79f-49cc-a2e4-154d85b79895" /></p>
   </details>

* **MILES: Visual BERT Pre-training with Injected Language Semantics for Video-text Retrieval**<br>
*Yuying Ge, Yixiao Ge, Xihui Liu, Alex Jinpeng Wang, Jianping Wu, Ying Shan, Xiaohu Qie, Ping Luo*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2204.12408)]
[[Code](https://github.com/tencentarc/mcq)]
   <details close>
   <summary>MILES Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/207725934-fc1c45cc-3946-4617-9eff-f93bd2903ba6.png" /></p>
   </details>

* **MAR: Masked Autoencoders for Efficient Action Recognition**<br>
*Zhiwu Qing, Shiwei Zhang, Ziyuan Huang, Xiang Wang, Yuehuan Wang, Yiliang Lv, Changxin Gao, Nong Sang*<br>
ArXiv'2022 [[Paper](http://arxiv.org/abs/2207.11660)]
   <details close>
   <summary>MAR Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/207726266-39eb361a-7e3c-4737-a51b-bfda8ad8ed06.png" /></p>
   </details>

* **Self-supervised Video Representation Learning with Motion-Aware Masked Autoencoders**<br>
*Haosen Yang, Deng Huang, Bin Wen, Jiannan Wu, Hongxun Yao, Yi Jiang, Xiatian Zhu, Zehuan Yuan*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2210.04154)]
[[Code](https://github.com/happy-hsy/MotionMAE)]
   <details close>
   <summary>MotionMAE Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/207726409-07a0255c-a7ae-4e6c-aa25-fdfa77393788.png" /></p>
   </details>

* **It Takes Two: Masked Appearance-Motion Modeling for Self-supervised Video Transformer Pre-training**<br>
*Yuxin Song, Min Yang, Wenhao Wu, Dongliang He, Fu Li, Jingdong Wang*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2210.05234)]
   <details close>
   <summary>MAM2 Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/234365532-88126c05-712d-4f1f-bba6-4d20a33a9c12.png" /></p>
   </details>

* **NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion**<br>
*Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, Nan Duan*<br>
ECCV'2022 [[Paper](https://arxiv.org/abs/2111.12417)]
[[Code](https://github.com/microsoft/NUWA)]
   <details close>
   <summary>NUWA Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/2358151d-7c19-441d-84fc-c1273b9e7fb0" /></p>
   </details>

* **MIMT: Masked Image Modeling Transformer for Video Compression**<br>
*Jinxi Xiang, Kuan Tian, Jun Zhang*<br>
ICLR'2023 [[Paper](https://openreview.net/forum?id=j9m-mVnndbm)]
   <details close>
   <summary>MIMT Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/207726629-e9481b07-58a4-4afb-be42-1a315bccd10c.png" /></p>
   </details>

* **VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking**<br>
*Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, Yu Qiao*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2303.16727)]
[[Code](https://github.com/MCG-NJU/VideoMAE)]
   <details close>
   <summary>VideoMAE.V2 Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/231277665-7027e34b-7b2c-4306-ac73-7be08b176f09.png" /></p>
   </details>

* **OmniMAE: Single Model Masked Pretraining on Images and Videos**<br>
*Rohit Girdhar, Alaaeldin El-Nouby, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra*<br>
CVPR'2023 [[Paper](http://arxiv.org/abs/2206.08356)]
[[Code](https://github.com/facebookresearch/omnivore)]
   <details close>
   <summary>OmniMAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/207725594-3beaf158-74f6-4fed-a330-d34755d1f37a.png" /></p>
   </details>

* **Masked Video Distillation: Rethinking Masked Feature Modeling for Self-supervised Video Representation Learning**<br>
*Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Lu Yuan, Yu-Gang Jiang*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2212.04500)]
[[Code](https://github.com/ruiwang2021/mvd)]
   <details close>
   <summary>MVD Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/26aeb76c-cfd1-47e4-b419-2298384aa6f9" /></p>
   </details>

* **DropMAE: Masked Autoencoders with Spatial-Attention Dropout for Tracking Tasks**<br>
*Qiangqiang Wu, Tianyu Yang, Ziquan Liu, Baoyuan Wu, Ying Shan, Antoni B. Chan*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2304.00571)]
[[Code](https://github.com/jimmy-dq/dropmae)]
   <details close>
   <summary>DropMAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/229926942-c661dff0-1cba-43cd-a206-435f223d8fd6.png" /></p>
   </details>

* **AdaMAE: Adaptive Masking for Efficient Spatiotemporal Learning with Masked Autoencoders**<br>
*Wele Gedara Chaminda Bandara, Naman Patel, Ali Gholami, Mehdi Nikkhah, Motilal Agrawal, Vishal M. Patel*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2211.09120)]
[[Code](https://github.com/wgcban/adamae)]
   <details close>
   <summary>AdaMAE Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/251232601-e06047d1-478a-4000-b6c5-7c52e0fae274.png" /></p>
   </details>

* **MAGVIT: Masked Generative Video Transformer**<br>
*Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, Lu Jiang*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2212.05199)]
[[Code](https://github.com/MAGVIT/magvit)]
   <details close>
   <summary>MAGVIT Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/234349205-1831ce49-2e39-440b-a9b6-916d831f0502.png" /></p>
   </details>

* **CMAE-V: Contrastive Masked Autoencoders for Video Action Recognition**<br>
*Cheng-Ze Lu, Xiaojie Jin, Zhicheng Huang, Qibin Hou, Ming-Ming Cheng, Jiashi Feng*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2301.06018)]
   <details close>
   <summary>CMAE-V Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/236318547-a6b49259-bcd5-4e76-b5a9-c20c19a65719.png" /></p>
   </details>

* **Siamese Masked Autoencoders**<br>
*Agrim Gupta, Jiajun Wu, Jia Deng, Li Fei-Fei*<br>
NeurIPS'2023 [[Paper](https://arxiv.org/abs/2305.14344)]
[[Code](https://siam-mae-video.github.io/)]
   <details close>
   <summary>SiamMAE Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/68a888de-dc34-4f5b-bdcc-b28dbfb99238" /></p>
   </details>

* **MGMAE: Motion Guided Masking for Video Masked Autoencoding**<br>
*Bingkun Huang, Zhiyu Zhao, Guozhen Zhang, Yu Qiao, Limin Wang*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2308.10794)]
[[Code](https://github.com/MCG-NJU/MGMAE)]
   <details close>
   <summary>MGMAE Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/263546126-988c91d2-1a00-4be6-aa4a-f1d3a4c428b7.png" /></p>
   </details>

* **Forecast-MAE: Self-supervised Pre-training for Motion Forecasting with Masked Autoencoders**<br>
*Jie Cheng, Xiaodong Mei, Ming Liu*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2308.09882)]
[[Code](https://github.com/jchengai/forecast-mae)]
   <details close>
   <summary>Forecast-MAE Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/263546126-988c91d2-1a00-4be6-aa4a-f1d3a4c428b7.png" /></p>
   </details>

* **Traj-MAE: Masked Autoencoders for Trajectory Prediction**<br>
*Hao Chen, Jiaze Wang, Kun Shao, Furui Liu, Jianye Hao, Chenyong Guan, Guangyong Chen, Pheng-Ann Heng*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2303.06697)]
   <details close>
   <summary>Traj-MAE Framework</summary>
   <p align="center"><img width="85%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273470265-f51fd9c3-b8c7-46f8-a253-79e0db2cb3c3.png" /></p>
   </details>

* **HumanMAC: Masked Motion Completion for Human Motion Prediction**<br>
*Ling-Hao Chen, Jiawei Zhang, Yewen Li, Yiren Pang, Xiaobo Xia, Tongliang Liu*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2308.07092)]
[[Code](https://github.com/linghaochan/humanmac)]
   <details close>
   <summary>HumanMAC Framework</summary>
   <p align="center"><img width="50%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273471835-fd4c7cd5-0bcc-4cc2-9eb7-bdb0a2892adc.png" /></p>
   </details>

* **SkeletonMAE: Graph-based Masked Autoencoder for Skeleton Sequence Pre-training**<br>
*Hong Yan, Yang Liu, Yushen Wei, Zhen Li, Guanbin Li, Liang Lin*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2307.08476)]
[[Code](https://github.com/HongYan1123/SkeletonMAE)]
   <details close>
   <summary>SkeletonMAE Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273470987-0094f1d9-5378-4f88-948d-e23818b10e19.png" /></p>
   </details>

* **Masked Motion Predictors are Strong 3D Action Representation Learners**<br>
*Ling-Hao Chen, Jiawei Zhang, Yewen Li, Yiren Pang, Xiaobo Xia, Tongliang Liu*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2302.03665)]
[[Code](https://github.com/maoyunyao/MAMP)]
   <details close>
   <summary>MAMP Framework</summary>
   <p align="center"><img width="85%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273513853-a2fac3d8-9f7c-4da7-ac36-d43c68d8becc.png" /></p>
   </details>

* **GeoMIM: Towards Better 3D Knowledge Transfer via Masked Image Modeling for Multi-view 3D Understanding**<br>
*Jihao Liu, Tai Wang, Boxiao Liu, Qihang Zhang, Yu Liu, Hongsheng Li*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2303.11325)]
[[Code](https://github.com/Sense-X/GeoMIM)]
   <details close>
   <summary>GeoMIM Framework</summary>
   <p align="center"><img width="85%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273514669-2236d35b-084c-46ea-82b0-913d5306d29d.png" /></p>
   </details>

* **Motion-Guided Masking for Spatiotemporal Representation Learning**<br>
*David Fan, Jue Wang, Shuai Liao, Yi Zhu, Vimal Bhat, Hector Santos-Villalobos, Rohith MV, Xinyu Li*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2308.12962)]
   <details close>
   <summary>MGM Framework</summary>
   <p align="center"><img width="85%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273517980-784780bd-40df-42d1-9768-d7f3e60f31ed.png" /></p>
   </details>

* **ModelScope Text-to-Video Technical Report**<br>
*Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, Shiwei Zhang*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2308.06571)]
[[Code](https://modelscope.cn/models/iic/text-to-video-synthesis/summary)]
   <details close>
   <summary>ModelScopeT2V Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/b9a4175d-f705-4932-b31e-df8f7aff1fc8" /></p>
   </details>

* **NUWA-Infinity: Autoregressive over Autoregressive Generation for Infinite Visual Synthesis**<br>
*Chenfei Wu, Jian Liang, Xiaowei Hu, Zhe Gan, Jianfeng Wang, Lijuan Wang, Zicheng Liu, Yuejian Fang, Nan Duan*<br>
NeurIPS'2023 [[Paper](https://arxiv.org/abs/2207.09814)]
[[Code](https://github.com/microsoft/NUWA)]
   <details close>
   <summary>NUWA-Infinity Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/d8223f34-37bb-4317-b8fd-77e74c0a6d12" /></p>
   </details>

* **NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation**<br>
*Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, Jianlong Fu, Gong Ming, Lijuan Wang, Zicheng Liu, Houqiang Li, Nan Duan*<br>
ACL'2023 [[Paper](https://arxiv.org/abs/2303.12346)]
[[Code](https://github.com/microsoft/NUWA)]
   <details close>
   <summary>NUWA-XL Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/798699d6-5f9c-461b-8da6-66b919c8872c" /></p>
   </details>

* **VDT: General-purpose Video Diffusion Transformers via Mask Modeling**<br>
*Haoyu Lu, Guoxing Yang, Nanyi Fei, Yuqi Huo, Zhiwu Lu, Ping Luo, Mingyu Ding*<br>
ICLR'2024 [[Paper](https://openreview.net/forum?id=Un0rgm9f04)]
[[Code](https://github.com/RERV/VDT)]
   <details close>
   <summary>VDT Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/6953a706-2c50-4146-b6db-420f790dd68d" /></p>
   </details>

<p align="right">(<a href="#top">back to top</a>)</p>

### Knowledge Distillation and Few-shot Classification

* **Generic-to-Specific Distillation of Masked Autoencoders**<br>
*Wei Huang, Zhiliang Peng, Li Dong, Furu Wei, Jianbin Jiao, Qixiang Ye*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2302.14771)]
[[Code](https://github.com/pengzhiliang/G2SD)]
   <details close>
   <summary>G2SD Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/229930409-40ca4ec9-68df-4eed-ae49-67f513a99277.png" /></p>
   </details>

* **Masked Autoencoders Enable Efficient Knowledge Distillers**<br>
*Yutong Bai, Zeyu Wang, Junfei Xiao, Chen Wei, Huiyu Wang, Alan Yuille, Yuyin Zhou, Cihang Xie*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2208.12256)]
[[Code](https://github.com/UCSC-VLAA/DMAE)]
   <details close>
   <summary>DMAE Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204308505-f730651b-6e24-4496-9c6d-879b2f446793.png" /></p>
   </details>

* **Mask-guided Vision Transformer (MG-ViT) for Few-Shot Learning**<br>
*Yuzhong Chen, Zhenxiang Xiao, Lin Zhao, Lu Zhang, Haixing Dai, David Weizhong Liu, Zihao Wu, Changhe Li, Tuo Zhang, Changying Li, Dajiang Zhu, Tianming Liu, Xi Jiang*<br>
ICLR'2023 [[Paper](http://arxiv.org/abs/2205.09995)]
   <details close>
   <summary>MG-ViT Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/39c153eb-5ff0-4777-bd90-7c8b9bc53c1d" /></p>
   </details>

* **Masked Autoencoders Are Stronger Knowledge Distillers**<br>
*Shanshan Lao, Guanglu Song, Boxiao Liu, Yu Liu, Yujiu Yang*<br>
ICCV'2023 [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lao_Masked_Autoencoders_Are_Stronger_Knowledge_Distillers_ICCV_2023_paper.pdf)]
   <details close>
   <summary>MKD Framework</summary>
   <p align="center"><img width="80%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273472136-59171feb-6a36-4027-8a21-14e17c230eb2.png" /></p>
   </details>

### Efficient Fine-tuning

* **Masked Images Are Counterfactual Samples for Robust Fine-tuning**<br>
*Yao Xiao, Ziyi Tang, Pengxu Wei, Cong Liu, Liang Lin*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2303.03052)]
[[Code](https://github.com/Coxy7/robust-finetuning)]
   <details close>
   <summary>Robust Finetuning Framework</summary>
   <p align="center"><img width="65%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/251233953-dfec8ab8-9c23-4395-9b67-e555b62b17ec.png" /></p>
   </details>

* **Contrastive Tuning: A Little Help to Make Masked Autoencoders Forget**<br>
*Johannes Lehner, Benedikt Alkin, Andreas Fürst, Elisabeth Rumetshofer, Lukas Miklautz, Sepp Hochreiter*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2304.10520)]
[[Code](https://github.com/ml-jku/MAE-CT)]
   <details close>
   <summary>MAE-CT Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/236308666-b5904b49-dd4e-4dc5-9ee2-158eb3e616e5.png" /></p>
   </details>

* **Masked Autoencoders are Efficient Class Incremental Learners**<br>
*Jiang-Tian Zhai, Xialei Liu, Andrew D. Bagdanov, Ke Li, Ming-Ming Cheng*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2308.12510)]
[[Code](https://github.com/scok30/mae-cil)]
   <details close>
   <summary>MAE-CIL Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273471437-29db5e9d-6f82-4c6f-80b5-83c78171c52c.png" /></p>
   </details>

* **MaskMatch: Boosting Semi-Supervised Learning Through Mask Autoencoder-Driven Feature Learning**<br>
*Wenjin Zhang, Keyi Li, Sen Yang, Chenyang Gao, Wanzhao Yang, Sifan Yuan, Ivan Marsic*<br>
arXiv'2024 [[Paper](https://arxiv.org/abs/2405.06227)]
   <details close>
   <summary>MaskMatch Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/151bf557-cea2-4a7a-8385-bc55dff07c6f" /></p>
   </details>

* **Pseudo Labelling for Enhanced Masked Autoencoders**<br>
*Srinivasa Rao Nandam, Sara Atito, Zhenhua Feng, Josef Kittler, Muhammad Awais*<br>
arXiv'2024 [[Paper](https://arxiv.org/abs/2406.17450)]
   <details close>
   <summary>SdAE Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Westlake-AI/openmixup/assets/44519745/7ddf82e3-73da-4bb2-a294-9d2dece750a6" /></p>
   </details>

### Medical Image

* **Self Pre-training with Masked Autoencoders for Medical Image Analysis**<br>
*Lei Zhou, Huidong Liu, Joseph Bae, Junjun He, Dimitris Samaras, Prateek Prasanna*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2203.05573)]

* **Self-distillation Augmented Masked Autoencoders for Histopathological Image Classification**<br>
*Yang Luo, Zhineng Chen, Xieping Gao*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2203.16983)]

* **Global Contrast Masked Autoencoders Are Powerful Pathological Representation Learners**<br>
*Hao Quan, Xingyu Li, Weixing Chen, Qun Bai, Mingchen Zou, Ruijie Yang, Tingting Zheng, Ruiqun Qi, Xinghua Gao, Xiaoyu Cui*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2205.09048)] [[Code](https://github.com/staruniversus/gcmae)]

* **FreMAE: Fourier Transform Meets Masked Autoencoders for Medical Image Segmentation**<br>
*Wenxuan Wang, Jing Wang, Chen Chen, Jianbo Jiao, Lichao Sun, Yuanxiu Cai, Shanshan Song, Jiangyun Li*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2304.10864)]

* **Masked Image Modeling Advances 3D Medical Image Analysis**<br>
*Zekai Chen, Devansh Agarwal, Kshitij Aggarwal, Wiem Safta, Samit Hirawat, Venkat Sethuraman, Mariann Micsinai Balan, Kevin Brown*<br>
WACV'2023 [[Paper](https://arxiv.org/abs/2204.11716)] [[Code](https://github.com/ZEKAICHEN/MIM-Med3D)]

* **MRM: Masked Relation Modeling for Medical Image Pre-Training with Genetics**<br>
*Qiushi Yang, Wuyang Li, Baopu Li, Yixuan Yuan*<br>
ICCV'2023 [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_MRM_Masked_Relation_Modeling_for_Medical_Image_Pre-Training_with_Genetics_ICCV_2023_paper.pdf)] [[Code](https://github.com/CityU-AIM-Group/MRM)]

* **FocusMAE: Gallbladder Cancer Detection from Ultrasound Videos with Focused Masked Autoencoders**<br>
*Soumen Basu, Mayuna Gupta, Chetan Madan, Pankaj Gupta, Chetan Arora*<br>
CVPR'2024 [[Paper](https://arxiv.org/abs/2403.08848)] [[Code](https://github.com/sbasu276/FocusMAE)]

### Face Recognition

* **FaceMAE: Privacy-Preserving Face Recognition via Masked Autoencoders**<br>
*Kai Wang, Bo Zhao, Xiangyu Peng, Zheng Zhu, Jiankang Deng, Xinchao Wang, Hakan Bilen, Yang You*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2205.11090)] [Code](https://github.com/kaiwang960112/FaceMAE)]

### Scene Text Recognition (OCR)

* **MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining**<br>
*Pengyuan Lyu, Chengquan Zhang, Shanshan Liu, Meina Qiao, Yangliu Xu, Liang Wu, Kun Yao, Junyu Han, Errui Ding, Jingdong Wang*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2206.00311)]

* **DiT: Self-supervised Pre-training for Document Image Transformer**<br>
*Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei*<br>
ACMMM'2022 [[Paper](https://arxiv.org/abs/2203.02378)] [Code](https://github.com/microsoft/unilm/tree/master/dit)]
[[Code](https://github.com/scok30/mae-cil)]
   <details close>
   <summary>DiT Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/1dfc20b5-44d0-437f-93ed-7dd3379a742f" /></p>
   </details>

* **DocMAE: Document Image Rectification via Self-supervised Representation Learning**<br>
*Shaokai Liu, Hao Feng, Wengang Zhou, Houqiang Li, Cong Liu, Feng Wu*<br>
ICME'2023 [[Paper](https://arxiv.org/abs/2304.10341)]

### Remote Sensing Image

* **SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery**<br>
*Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David B. Lobell, Stefano Ermon*<br>
NeurIPS'2022 [[Paper](https://arxiv.org/abs/2207.08051)]

* **CMID: A Unified Self-Supervised Learning Framework for Remote Sensing Image Understanding**<br>
*Dilxat Muhtar, Xueliang Zhang, Pengfeng Xiao, Zhenshi Li, Feng Gu*<br>
TGRS'2023 [[Paper](https://arxiv.org/abs/2304.09670)] [[Code](https://github.com/NJU-LHRS/official-CMID)]

* **Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning**<br>
*Colorado J Reed, Ritwik Gupta, Shufan Li, Sarah Brockman, Christopher Funk, Brian Clipp, Kurt Keutzer, Salvatore Candido, Matt Uyttendaele, Trevor Darrell*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2212.14532)]

* **SS-MAE: Spatial-Spectral Masked Auto-Encoder for Multi-Source Remote Sensing Image Classification**<br>
*Junyan Lin, Feng Gao, Xiaocheng Shi, Junyu Dong, Qian Du*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2311.04442)]

* **On the Transferability of Learning Models for Semantic Segmentation for Remote Sensing Data**<br>
*Rongjun Qin, Guixiang Zhang, Yang Tang*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2310.10490)]

* **Fus-MAE: A cross-attention-based data fusion approach for Masked Autoencoders in remote sensing**<br>
*Hugo Chan-To-Hing, Bharadwaj Veeravalli*<br>
ArXiv'2024 [[Paper](https://arxiv.org/abs/2401.02764)]

* **S2MAE: A Spatial-Spectral Pretraining Foundation Model for Spectral Remote Sensing Data**<br>
*Xuyang Li, Danfeng Hong, Jocelyn Chanussot*<br>
CVPR'2024 [[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Li_S2MAE_A_Spatial-Spectral_Pretraining_Foundation_Model_for_Spectral_Remote_Sensing_CVPR_2024_paper.html)] 

### 3D Representation Learning

* **Pre-Training 3D Point Cloud Transformers with Masked Point Modeling**<br>
*Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, Jiwen Lu*<br>
CVPR'2022 [[Paper](https://arxiv.org/abs/2111.14819)] [[Code](https://github.com/lulutang0608/Point-BERT)]

* **Masked Autoencoders for Point Cloud Self-supervised Learning**<br>
*Yatian Pang, Wenxiao Wang, Francis E.H. Tay, Wei Liu, Yonghong Tian, Li Yuan*<br>
ECCV'2022 [[Paper](https://arxiv.org/abs/2203.06604)] [[Code](https://github.com/Pang-Yatian/Point-MAE)]

* **Masked Discrimination for Self-Supervised Learning on Point Clouds**<br>
*Haotian Liu, Mu Cai, Yong Jae Lee*<br>
ECCV'2022 [[Paper](https://arxiv.org/abs/2203.11183)] [[Code](https://github.com/haotian-liu/MaskPoint)]

* **MeshMAE: Masked Autoencoders for 3D Mesh Data Analysis**<br>
*Yaqian Liang, Shanshan Zhao, Baosheng Yu, Jing Zhang, Fazhi He*<br>
ECCV'2022 [[Paper](http://arxiv.org/abs/2207.10228)]

* **Voxel-MAE: Masked Autoencoders for Pre-training Large-scale Point Clouds**<br>
*Chen Min, Xinli Xu, Dawei Zhao, Liang Xiao, Yiming Nie, Bin Dai*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2206.09900)]

* **Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training**<br>
*Renrui Zhang, Ziyu Guo, Peng Gao, Rongyao Fang, Bin Zhao, Dong Wang, Yu Qiao, Hongsheng Li*<br>
NeurIPS'2022 [[Paper](https://arxiv.org/abs/2205.14401)]

* **Ponder: Point Cloud Pre-training via Neural Rendering**<br>
*Renrui Zhang, Ziyu Guo, Peng Gao, Rongyao Fang, Bin Zhao, Dong Wang, Yu Qiao, Hongsheng Li*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2301.00157)] [[Code](https://dihuangdh.github.io/ponder)]

* **Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders**<br>
*Renrui Zhang, Liuhui Wang, Yu Qiao, Peng Gao, Hongsheng Li*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2212.06785)] [[Code](https://github.com/zrrskywalker/point-m2ae)]

* **GeoMAE: Masked Geometric Target Prediction for Self-supervised Point Cloud Pre-Training**<br>
*Xiaoyu Tian, Haoxi Ran, Yue Wang, Hang Zhao*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2305.08808)] [[Code](https://github.com/ZrrSkywalker/I2P-MAE)]

* **VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion**<br>
*Yiming Li, Zhiding Yu, Christopher Choy, Chaowei Xiao, Jose M. Alvarez, Sanja Fidler, Chen Feng, Anima Anandkumar*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2302.12251)] [[Code](https://github.com/NVlabs/VoxFormer)]

* **Autoencoders as Cross-Modal Teachers: Can Pretrained 2D Image Transformers Help 3D Representation Learning?**<br>
*Runpei Dong, Zekun Qi, Linfeng Zhang, Junbo Zhang, Jianjian Sun, Zheng Ge, Li Yi, Kaisheng Ma*<br>
ICLR'2023 [[Paper](https://arxiv.org/abs/2212.08320)] [[Code](https://github.com/tsinghua-mars-lab/geomae)]

* **Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining**<br>
*Zekun Qi, Runpei Dong, Guofan Fan, Zheng Ge, Xiangyu Zhang, Kaisheng Ma, Li Yi*<br>
ICML'2023 [[Paper](https://arxiv.org/abs/2302.02318)] [[Code](https://github.com/qizekun/ReCon)]

* **MGM: A meshfree geometric multilevel method for systems arising from elliptic equations on point cloud surfaces**<br>
*Grady B. Wright, Andrew M. Jones, Varun Shankar*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2204.06154)]

* **PointGPT: Auto-regressively Generative Pre-training from Point Clouds**<br>
*Guangyan Chen, Meiling Wang, Yi Yang, Kai Yu, Li Yuan, Yufeng Yue*<br>
NeurIPS'2023 [[Paper](https://arxiv.org/abs/2305.11487)] [[Code](https://github.com/CGuangyan-BIT/PointGPT)]

* **MATE: Masked Autoencoders are Online 3D Test-Time Learners**<br>
*M. Jehanzeb Mirza, Inkyu Shin, Wei Lin, Andreas Schriebl, Kunyang Sun, Jaesung Choe, Horst Possegger, Mateusz Kozinski, In So Kweon, Kun-Jin Yoon, Horst Bischof*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2211.11432)] [[Code](https://github.com/jmiemirza/MATE)]

* **Masked Spatio-Temporal Structure Prediction for Self-supervised Learning on Point Cloud Videos**<br>
*Zhiqiang Shen, Xiaoxiao Sheng, Hehe Fan, Longguang Wang, Yulan Guo, Qiong Liu, Hao Wen, Xi Zhou*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2308.09245)] [[Code](https://github.com/JohnsonSign/MaST-Pre)]
   <details close>
   <summary>MaST-Pre</summary>
   <p align="center"><img width="70%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273511030-25d49434-d197-42fa-a11a-2ce02458b938.png" /></p>
   </details>

* **UniPAD: A Universal Pre-training Paradigm for Autonomous Driving**<br>
*Honghui Yang, Sha Zhang, Di Huang, Xiaoyang Wu, Haoyi Zhu, Tong He, Shixiang Tang, Hengshuang Zhao, Qibo Qiu, Binbin Lin, Xiaofei He, Wanli Ouyang*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2310.08370)] [[Code](https://github.com/Nightmare-n/UniPAD)]
   <details close>
   <summary>UniPAD</summary>
   <p align="center"><img width="85%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/275272898-a2f31d20-f405-4733-a0b9-72b34c8e0525.png" /></p>
   </details>

* **PonderV2: Pave the Way for 3D Foundataion Model with A Universal Pre-training Paradigm**<br>
*Haoyi Zhu, Honghui Yang, Xiaoyang Wu, Di Huang, Sha Zhang, Xianglong He, Tong He, Hengshuang Zhao, Chunhua Shen, Yu Qiao, Wanli Ouyang*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2310.08586)]
[[Code](https://github.com/Pointcept/Pointcept)]

* **NeRF-MAE : Masked AutoEncoders for Self Supervised 3D representation Learning for Neural Radiance Fields**<br>
*Muhammad Zubair Irshad, Sergey Zakahrov, Vitor Guizilini, Adrien Gaidon, Zsolt Kira, Rares Ambrus*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2404.01300)]
[[Code](https://github.com/zubair-irshad/NeRF-MAE)]

* **General Point Model with Autoencoding and Autoregressive**<br>
*Zhe Li, Zhangyang Gao, Cheng Tan, Bocheng Ren, Laurence Tianruo Yang, Stan Z. Li*<br>
CVPR'2024 [[Paper](https://arxiv.org/abs/2310.16861)]
[[Code](https://github.com/gentlefress/GPM)]

### Low-level Vision

* **DegAE: A New Pretraining Paradigm for Low-level Vision**<br>
*Yihao Liu, Jingwen He, Jinjin Gu, Xiangtao Kong, Yu Qiao, Chao Dong*<br>
CVPR'2023 [[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_DegAE_A_New_Pretraining_Paradigm_for_Low-Level_Vision_CVPR_2023_paper.html)]
[[Code](https://github.com/lyh-18/DegAE_DegradationAutoencoder/)]

* **LM4LV: A Frozen Large Language Model for Low-level Vision Tasks**<br>
*Boyang Zheng, Jinjin Gu, Shijun Li, Chao Dong*<br>
ArXiv'2024 [[Paper](https://arxiv.org/abs/2405.15734)]
[[Code](https://github.com/bytetriper/LM4LV)]

### Depth Estimation

* **MeSa: Masked, Geometric, and Supervised Pre-training for Monocular Depth Estimation**<br>
*Muhammad Osama Khan, Junbang Liang, Chun-Kai Wang, Shan Yang, Yu Lou*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2310.04551)]
   <details close>
   <summary>UniPAD</summary>
   <p align="center"><img width="85%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/275282765-21d62c23-9b2f-442c-8cbb-be12d283393d.png" /></p>
   </details>

<p align="right">(<a href="#top">back to top</a>)</p>

## Audio and Speech

* **wav2vec: Unsupervised Pre-training for Speech Recognition**<br>
*Steffen Schneider, Alexei Baevski, Ronan Collobert, Michael Auli*<br>
ArXiv'2019 [[Paper](https://arxiv.org/abs/1904.05862)] [[Code](https://github.com/pytorch/fairseq)]

* **vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations**<br>
*Alexei Baevski, Steffen Schneider, Michael Auli*<br>
ArXiv'2019 [[Paper](https://arxiv.org/abs/1910.05453)] [[Code](https://github.com/pytorch/fairseq)]

* **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**<br>
*Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli*<br>
NeurIPS'2020 [[Paper](https://arxiv.org/abs/2006.11477)] [[Code](https://github.com/pytorch/fairseq)]

* **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units**<br>
*Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed*<br>
TASLP'2021 [[Paper](https://arxiv.org/abs/2106.07447)] [[Code](https://github.com/pytorch/fairseq)]

* **MAM: Masked Acoustic Modeling for End-to-End Speech-to-Text Translation**<br>
*Junkun Chen, Mingbo Ma, Renjie Zheng, Liang Huang*<br>
ArXiv'2021 [[Paper](https://arxiv.org/abs/2010.11445)]

* **MAE-AST: Masked Autoencoding Audio Spectrogram Transformer**<br>
*Alan Baade, Puyuan Peng, David Harwath*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2203.16691)] [[Code](https://github.com/AlanBaade/MAE-AST-Public)]

* **Masked Spectrogram Prediction For Self-Supervised Audio Pre-Training**<br>
*Dading Chong, Helin Wang, Peilin Zhou, Qingcheng Zeng*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2204.12768)] [[Code](https://github.com/wanghelin1997/maskspec)]

* **Masked Autoencoders that Listen**<br>
*Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, Christoph Feichtenhofer*<br>
NeurIPS'2022 [[Paper](https://arxiv.org/abs/2207.06405)] [[Code](https://github.com/facebookresearch/audiomae)]

* **Contrastive Audio-Visual Masked Autoencoder**<br>
*Yuan Gong, Andrew Rouditchenko, Alexander H. Liu, David Harwath, Leonid Karlinsky, Hilde Kuehne, James Glass*<br>
ICLR'2023 [[Paper](https://arxiv.org/abs/2210.07839)]

* **Audiovisual Masked Autoencoders**<br>
*Mariana-Iuliana Georgescu, Eduardo Fonseca, Radu Tudor Ionescu, Mario Lucic, Cordelia Schmid, Anurag Arnab*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2210.07839)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="85%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/273470692-dc127f79-2fd4-4aff-a911-a7845c58ed1b.png" /></p>
   </details>

* **Masked Autoencoders with Multi-Window Local-Global Attention Are Better Audio Learners**<br>
*Sarthak Yadav, Sergios Theodoridis, Lars Kai Hansen, Zheng-Hua Tan*<br>
ICLR'2024 [[Paper](https://arxiv.org/abs/2306.00561)]

* **Masked Audio Generation using a Single Non-Autoregressive Transformer**<br>
*Alon Ziv, Itai Gat, Gael Le Lan, Tal Remez, Felix Kreuk, Jade Copet, Alexandre Défossez, Gabriel Synnaeve, Yossi Adi*<br>
ICLR'2024 [[Paper](https://openreview.net/forum?id=Ny8NiVfi95)]
[[Code](https://pages.cs.huji.ac.il/adiyoss-lab/MAGNeT/)]

## AI for Science

### Protein

* **Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences**<br>
*Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, Rob Fergus*<br>
PNAS'2020 [[Paper](https://www.pnas.org/doi/pdf/10.1073/pnas.2016239118)]
[[Code](https://github.com/facebookresearch/esm)]

* **Transformer protein language models are unsupervised structure learners**<br>
*Roshan Rao, Joshua Meier, Tom Sercu, Sergey Ovchinnikov, Alexander Rives*<br>
bioRxiv'2020 [[Paper](https://www.biorxiv.org/content/10.1101/2020.12.15.422761)]
[[Code](https://github.com/facebookresearch/esm)]

* **Language models enable zero-shot prediction of the effects of mutations on protein function**<br>
*Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, Alexander Rives*<br>
bioRxiv'2021 [[Paper](https://www.biorxiv.org/content/10.1101/2022.04.10.487779)]
[[Code](https://github.com/facebookresearch/esm)]

* **Learning inverse folding from millions of predicted structures**<br>
*Chloe Hsu, Robert Verkuil, Jason Liu, Zeming Lin, Brian Hie, Tom Sercu, Adam Lerer, Alexander Rives*<br>
ICML'2022 [[Paper](https://www.biorxiv.org/content/10.1101/2022.04.10.487779)]
[[Code](https://github.com/facebookresearch/esm)]

* **Evolutionary-scale prediction of atomic level protein structure with a language model**<br>
*Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Salvatore Candido, Alexander Rives*<br>
bioRxiv'2022 [[Paper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)]
[[Code](https://github.com/facebookresearch/esm)]

* **ProteinBERT: A universal deep-learning model of protein sequence and function**<br>
*Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Salvatore Candido, Alexander Rives*<br>
Bioinformatics'2022 [[Paper](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274)]
[[Code](https://github.com/nadavbra/protein_bert)]

* **Foldseek: fast and accurate protein structure search**<br>
*Michel van Kempen, Stephanie S. Kim, Charlotte Tumescheit, Milot Mirdita, Johannes Söding, Martin Steinegger*<br>
Nature'2023 [[Paper](https://www.biorxiv.org/content/10.1101/2022.02.07.479398v2)]
[[Code](https://github.com/steineggerlab/foldseek)]

* **SaProt: Protein Language Modeling with Structure-aware Vocabulary**<br>
*Jin Su, Chenchen Han, Yuyang Zhou, Junjie Shan, Xibin Zhou, Fajie Yuan*<br>
ICLR'2024 [[Paper](https://www.biorxiv.org/content/10.1101/2023.10.01.560349v2)]
[[Code](https://github.com/westlake-repl/SaProt)]

* **MAPE-PPI: Towards Effective and Efficient Protein-Protein Interaction Prediction via Microenvironment-Aware Protein Embedding**<br>
*Lirong Wu, Yijun Tian, Yufei Huang, Siyuan Li, Haitao Lin, Nitesh V Chawla, Stan Z. Li*<br>
ICLR'2024 [[Paper](https://arxiv.org/abs/2402.14391)]
[[Code](https://github.com/lirongwu/mape-ppi)]

* **VQDNA: Unleashing the Power of Vector Quantization for Multi-Species Genomic Sequence Modeling**<br>
*Siyuan Li, Zedong Wang, Zicheng Liu, Di Wu, Cheng Tan, Jiangbin Zheng, Yufei Huang, Stan Z. Li*<br>
ICML'2024 [[Paper](https://arxiv.org/abs/2405.10812)]

* **Learning to Predict Mutation Effects of Protein-Protein Interactions by Microenvironment-aware Hierarchical Prompt Learning**<br>
*Lirong Wu, Yijun Tian, Haitao Lin, Yufei Huang, Siyuan Li, Nitesh V Chawla, Stan Z. Li*<br>
ICML'2024 [[Paper](https://arxiv.org/abs/2405.10348)]
[[Code](https://github.com/lirongwu/prompt-ddg)]

### Chemistry

* **Mole-BERT: Rethinking Pre-training Graph Neural Networks for Molecules**<br>
*Jun Xia, Chengshuai Zhao, Bozhen Hu, Zhangyang Gao, Cheng Tan, Yue Liu, Siyuan Li, Stan Z. Li*<br>
ICLR'2023 [[Paper](https://openreview.net/forum?id=jevY-DtiZTR)]
[[Code](https://github.com/junxia97/Mole-BERT)]

* **VQMAE: Surface-VQMAE: Vector-quantized Masked Auto-encoders on Molecular Surfaces**<br>
*Fang Wu, Stan Z. Li*<br>
ICML'2024 [[Paper](https://github.com/smiles724/VQMAE)]

### Physics

* **W-MAE: Pre-trained weather model with masked autoencoder for multi-variable weather forecasting**<br>
*Xin Man, Chenghong Zhang, Jin Feng, Changyu Li, Jie Shao*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2304.08754)]
[[Code](https://github.com/gufrannn/w-mae)]

* **Masked Autoencoders are PDE Learners**<br>
*Anthony Zhou, Amir Barati Farimani*<br>
arXiv'2024 [[Paper](https://arxiv.org/abs/2403.17728)]

<p align="right">(<a href="#top">back to top</a>)</p>

## Time Series and Neuroscience Learning

* **Neuro-BERT: Rethinking Masked Autoencoding for Self-Supervised Neurological Pretraining**<br>
*Di Wu, Siyuan Li, Jie Yang, Mohamad Sawan*<br>
JBHI'2024 [[Paper](https://doi.org/10.1109/JBHI.2024.3415959)]
[[Code](https://github.com/Westlake-AI/OpenBioSeq)]
   <details close>
   <summary>Neuro-BERT (neuro2vec)</summary>
   <p align="center"><img width="85%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/50e515cd-3c51-4f8c-b5e6-b8f8eea1aed9" /></p>
   </details>

* **Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI**<br>
*Wei-Bang Jiang, Li-Ming Zhao, Bao-Liang Lu*<br>
ICLR'2024 [[Paper](https://openreview.net/forum?id=QzTpTRVtrP)]
   <details close>
   <summary>LaBraM</summary>
   <p align="center"><img width="95%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/c651a84b-d7e6-48c6-81ae-1eff065ad418" /></p>
   </details>

* **Neuroformer: Multimodal and Multitask Generative Pretraining for Brain Data**<br>
*Antonis Antoniades, Yiyi Yu, Joseph Canzano, William Wang, Spencer LaVere Smith*<br>
ICLR'2024 [[Paper](https://arxiv.org/abs/2311.00136)]
[[Code](https://github.com/woanderer/Neuroformer)]
   <details close>
   <summary>Neuroformer</summary>
   <p align="center"><img width="75%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/14e4d77e-f58a-4db6-ab6d-5de8da5d2a87" /></p>
   </details>

* **VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters**<br>
*Mouxiang Chen, Lefei Shen, Zhuo Li, Xiaoyun Joy Wang, Jianling Sun, Chenghao Liu*<br>
ArXiv'2024 [[Paper](https://arxiv.org/abs/2408.17253)]
[[Code](https://github.com/keytoyze/visionts)]

## Reinforcement Learning

* **Mask-based Latent Reconstruction for Reinforcement Learning**<br>
*Tao Yu, Zhizheng Zhang, Cuiling Lan, Yan Lu, Zhibo Chen*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2201.12096)]

* **Masked Contrastive Representation Learning for Reinforcement Learning**<br>
*Jinhua Zhu, Yingce Xia, Lijun Wu, Jiajun Deng, Wengang Zhou, Tao Qin, Tie-Yan Liu, Houqiang Li*<br>
TPAMI'2023 [[Paper](https://ieeexplore.ieee.org/document/9779589)]
[[Code](https://github.com/teslacool/m-curl)]

* **SMART: Self-supervised Multi-task pretrAining with contRol Transformers**<br>
*Yanchao Sun, Shuang Ma, Ratnesh Madaan, Rogerio Bonatti, Furong Huang, Ashish Kapoor*<br>
ICLR'2023 [[Paper](https://arxiv.org/abs/2301.09816)]

<p align="right">(<a href="#top">back to top</a>)</p>

## Tabular Data

* **ReMasker: Imputing Tabular Data with Masked Autoencoding**<br>
*Tianyu Du, Luca Melis, Ting Wang*<br>
ICLR'2024 [[Paper](https://openreview.net/forum?id=KI9NqjLVDT)]

* **MCM: Masked Cell Modeling for Anomaly Detection in Tabular Data**<br>
*Jiaxin Yin, Yuanyuan Qiao, Zitang Zhou, Xiangchao Wang, Jie Yang*<br>
ICLR'2024 [[Paper](https://openreview.net/forum?id=lNZJyEDxy4)]

## Analysis and Understanding of Masked Modeling

* **Demystifying Self-Supervised Learning: An Information-Theoretical Framework**<br>
*Yao-Hung Hubert Tsai, Yue Wu, Ruslan Salakhutdinov, Louis-Philippe Morency*<br>
ICLR'2021 [[Paper](https://arxiv.org/abs/2006.05576)]

* **A Mathematical Exploration of Why Language Models Help Solve Downstream Tasks**<br>
*Nikunj Saunshi, Sadhika Malladi, Sanjeev Arora*<br>
ICLR'2021 [[Paper](https://arxiv.org/abs/2010.03648)]

* **Predicting What You Already Know Helps: Provable Self-Supervised Learning**<br>
*Jason D. Lee, Qi Lei, Nikunj Saunshi, Jiacheng Zhuo*<br>
NeurIPS'2021 [[Paper](https://arxiv.org/abs/2008.01064)]

* **How to Understand Masked Autoencoders**<br>
*Shuhao Cao, Peng Xu, David A. Clifton*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2202.03670)]

* **Masked prediction tasks: a parameter identifiability view**<br>
*Bingbin Liu, Daniel Hsu, Pradeep Ravikumar, Andrej Risteski*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2202.09305)]

* **Revealing the Dark Secrets of Masked Image Modeling**<br>
*Zhenda Xie, Zigang Geng, Jingcheng Hu, Zheng Zhang, Han Hu, Yue Cao*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2205.13543)]

* **Architecture-Agnostic Masked Image Modeling - From ViT back to CNN**<br>
*Siyuan Li, Di Wu, Fang Wu, Zelin Zang, Kai Wang, Lei Shang, Baigui Sun, Hao Li, Stan.Z.Li*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2205.13943)]

* **On Data Scaling in Masked Image Modeling**<br>
*Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Yixuan Wei, Qi Dai, Han Hu*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2206.04664)]

* **Towards Understanding Why Mask-Reconstruction Pretraining Helps in Downstream Tasks**<br>
*Jiachun Pan, Pan Zhou, Shuicheng Yan*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2206.03826)]

* **An Empirical Study Of Self-supervised Learning Approaches For Object Detection With Transformers**<br>
*Gokul Karthik Kumar, Sahal Shaji Mullappilly, Abhishek Singh Gehlot*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2205.05543)] [[Code](https://github.com/gokulkarthik/deformable-detr)]

* **Understanding Masked Image Modeling via Learning Occlusion Invariant Feature**<br>
*Xiangwen Kong, Xiangyu Zhang*<br>
ArXiv'2022 [[Paper](http://arxiv.org/abs/2208.04164)] [[Code](https://github.com/zhangq327/u-mae)]

* **How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders**<br>
*Qi Zhang, Yifei Wang, Yisen Wang*<br>
NeurIPS'2022 [[Paper](https://arxiv.org/abs/2210.08344)] [[Code](https://github.com/zhangq327/U-MAE)]

* **i-MAE: Are Latent Representations in Masked Autoencoders Linearly Separable**<br>
*Kevin Zhang, Zhiqiang Shen*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2210.11470)]

* **Understanding Masked Autoencoders via Hierarchical Latent Variable Models**<br>
*Lingjing Kong, Martin Q. Ma, Guangyi Chen, Eric P. Xing, Yuejie Chi, Louis-Philippe Morency, Kun Zhang*<br>
CVPR'2023 [[Paper](https://arxiv.org/abs/2306.04898)] [[Code](https://github.com/martinmamql/mae_understand)]

* **Evaluating Self-Supervised Learning via Risk Decomposition**<br>
*Yann Dubois, Tatsunori Hashimoto, Percy Liang*<br>
ICML'2023 [[Paper](https://arxiv.org/abs/2302.03068)] [[Code](https://github.com/yanndubs/ssl-risk-decomposition)]

* **Regeneration Learning: A Learning Paradigm for Data Generation**<br>
*Xu Tan, Tao Qin, Jiang Bian, Tie-Yan Liu, Yoshua Bengio*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2301.08846)]

<p align="right">(<a href="#top">back to top</a>)</p>

## Survey

* **A Survey on Masked Autoencoder for Self-supervised Learning in Vision and Beyond**<br>
*Chaoning Zhang, Chenshuang Zhang, Junha Song, John Seon Keun Yi, Kang Zhang, In So Kweon*<br>
IJCAI'2023 [[Paper](https://arxiv.org/abs/2208.00173)]

* **Masked Autoencoders in Computer Vision: A Comprehensive Survey**<br>
*Zexian Zhou, Xiaojing Liu*<br>
IEEE Access'2023 [[Paper](https://ieeexplore.ieee.org/document/10278410)]

* **Masked Modeling for Self-supervised Representation Learning on Vision and Beyond**<br>
*Siyuan Li, Luyuan Zhang, Zedong Wang, Di Wu, Lirong Wu, Zicheng Liu, Jun Xia, Cheng Tan, Yang Liu, Baigui Sun, Stan Z. Li*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2401.00897)] [[Code](https://github.com/Lupin1998/Awesome-MIM)]

## Contribution

Feel free to send [pull requests](https://github.com/Lupin1998/Awesome-MIM/pulls) to add more links with the following Markdown format. Note that the abbreviation, the code link, and the figure link are optional attributes.

```markdown
* **TITLE**<br>
*AUTHER*<br>
PUBLISH'YEAR [[Paper](link)] [[Code](link)]
   <details close>
   <summary>ABBREVIATION Framework</summary>
   <p align="center"><img width="90%" src="link_to_image" /></p>
   </details>
```

## Citation

If you find this repository and our survey helpful, please consider citing our paper:
```
@article{Li2023MIMSurvey,
  title={Masked Modeling for Self-supervised Representation Learning on Vision and Beyond},
  author={Siyuan Li and Luyuan Zhang and Zedong Wang and Di Wu and Lirong Wu and Zicheng Liu and Jun Xia and Cheng Tan and Yang Liu and Baigui Sun and Stan Z. Li},
  journal={ArXiv},
  year={2023},
  volume={abs/2401.00897},
}
```

## Related Project

### Paper List of Masked Image Modeling

- [Awesome-Masked-Autoencoders](https://github.com/EdisonLeeeee/Awesome-Masked-Autoencoders): A collection of literature after or concurrent with Masked Autoencoder (MAE).
- [awesome-MIM](https://github.com/ucasligang/awesome-MIM): Reading list for research topics in Masked Image Modeling.
- [Awesome-MIM](https://github.com/Westlake-AI/openmixup/blob/main/docs/en/awesome_selfsup/MIM.md): Awesome list of masked image modeling methods for self-supervised visual representation.
- [awesome-self-supervised-learning](https://github.com/jason718/awesome-self-supervised-learning): A curated list of awesome self-supervised methods.

### Project of Self-supervised Learning

- [unilm](https://github.com/microsoft/unilm): Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities.
- [OpenMixup](https://github.com/Westlake-AI/openmixup): CAIRI Supervised, Semi- and Self-Supervised Visual Representation Learning Toolbox and Benchmark.
- [MMPretrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab self-supervised pre-training toolbox and benchmark.
- [solo-learn](https://github.com/vturrisi/solo-learn): A library of self-supervised methods for visual representation learning powered by Pytorch Lightning.
- [VISSL](https://github.com/facebookresearch/vissl): FAIR's library of extensible, modular and scalable components for SOTA Self-Supervised Learning with images.
- [lightly](https://github.com/lightly-ai/lightly): A python library for self-supervised learning on images.
- [Fairseq](https://github.com/facebookresearch/fairseq): Facebook AI Research Sequence-to-Sequence Toolkit written in Python.

<p align="right">(<a href="#top">back to top</a>)</p>
