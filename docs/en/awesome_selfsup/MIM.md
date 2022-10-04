# Awesome Masked Image Modeling for Visual Represention

 [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![GitHub stars](https://img.shields.io/github/stars/Westlake-AI/openmixup?color=green) ![visitors](https://visitor-badge.glitch.me/badge?page_id=Westlake-AI/openmixup)

**We summarize masked image modeling (MIM) methods proposed for self-supervised visual representation learning.**
The list of awesome MIM methods is summarized in chronological order and is on updating.

**Table of Contents**

  - [MIM for Backbones](#mim-for-backbones)
    - [Fundermental Methods](#fundermental-methods)
    - [MIM with Constrastive Learning](#mim-with-constrastive-learning)
    - [MIM for Transformer and CNN](#mim-for-transformer-and-cnn)
    - [MIM with Advanced Masking](#mim-with-advanced-masking)
  - [MIM for Downstream Tasks](#mim-for-downstream-tasks)
    - [Object Detection](#object-detection)
    - [Video Rrepresentation](#video-rrepresentation)
    - [Medical Image](#medical-image)
    - [Face Recognition](#face-recognition)
    - [Scene Text Recognition (OCR)](#scene-text-recognition-ocr)
    - [Satellite Imagery](#satellite-imagery)
    - [3D Point Cloud](#3d-point-cloud)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Audio](#audio)
  - [Analysis of MIM](#analysis-of-mim)
  - [Contribution](#contribution)


## MIM for Backbones

### Fundermental Methods

* **iGPT**: Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, David Luan, Ilya Sutskever.
   - Generative Pretraining from Pixels. [[ICML'2020](http://proceedings.mlr.press/v119/chen20s/chen20s.pdf)] [[code](https://github.com/openai/image-gpt)]
* **ViT**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
   - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. [[ICLR'2021](https://arxiv.org/abs/2010.11929)] [[code](https://github.com/google-research/vision_transformer)]
* **BEiT**: Hangbo Bao, Li Dong, Furu Wei.
   - BEiT: BERT Pre-Training of Image Transformers. [[ICLR'2022](https://arxiv.org/abs/2106.08254)] [[code](https://github.com/microsoft/unilm/tree/master/beit)]
* **iBOT**: Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, Tao Kong.
   - iBOT: Image BERT Pre-Training with Online Tokenizer. [[ICLR'2022](https://arxiv.org/abs/2111.07832)] [[code](https://github.com/bytedance/ibot)]
* **MAE**: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick.
   - Masked Autoencoders Are Scalable Vision Learners. [[CVPR'2022](https://arxiv.org/abs/2111.06377)] [[code](https://github.com/facebookresearch/mae)]
* **SimMIM**: Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, Han Hu.
   - SimMIM: A Simple Framework for Masked Image Modeling. [[CVPR'2022](https://arxiv.org/abs/2111.09886)] [[code](https://github.com/microsoft/simmim)]
* **MaskFeat**: Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, Christoph Feichtenhofer.
   - Masked Feature Prediction for Self-Supervised Visual Pre-Training. [[CVPR'2022](https://arxiv.org/abs/2112.09133)] [[code](https://github.com/facebookresearch/SlowFast)]
* **data2vec**: Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli.
   - data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language. [[ICML'2022](https://arxiv.org/abs/2202.03555)] [[code](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)]
* **PeCo**: Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu.
   - PeCo: Perceptual Codebook for BERT Pre-training of Vision Transformers. [[ArXiv'2021](https://arxiv.org/abs/2111.12710)] [[code](https://github.com/microsoft/PeCo)]
* **MC-SSL0.0**: Sara Atito, Muhammad Awais, Ammarah Farooq, Zhenhua Feng, Josef Kittler.
   - MC-SSL0.0: Towards Multi-Concept Self-Supervised Learning. [[ArXiv'2021](https://arxiv.org/abs/2111.15340)]
* **mc-BEiT**: Xiaotong Li, Yixiao Ge, Kun Yi, Zixuan Hu, Ying Shan, Ling-Yu Duan.
   - mc-BEiT: Multi-choice Discretization for Image BERT Pre-training. [[ECCV'2022](https://arxiv.org/abs/2203.15371)] [[code](https://github.com/lixiaotong97/mc-BEiT)]
* **BootMAE**: Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu.
   - Bootstrapped Masked Autoencoders for Vision BERT Pretraining. [[ECCV'2022](https://arxiv.org/abs/2207.07116)] [[code](https://github.com/LightDXY/BootMAE)]
* **SdAE**: Yabo Chen, Yuchen Liu, Dongsheng Jiang, Xiaopeng Zhang, Wenrui Dai, Hongkai Xiong, Qi Tian.
   - SdAE: Self-distillated Masked Autoencoder. [[ECCV'2022](https://arxiv.org/abs/2208.00449)] [[code](https://github.com/AbrahamYabo/SdAE)]
* **SupMAE**: Feng Liang, Yangguang Li, Diana Marculescu.
   - SupMAE: Supervised Masked Autoencoders Are Efficient Vision Learners. [[ArXiv'2022](https://arxiv.org/abs/2205.14540)] [[code](https://github.com/cmu-enyac/supmae)]
* **MVP**: Longhui Wei, Lingxi Xie, Wengang Zhou, Houqiang Li, Qi Tian.
   - MVP: Multimodality-guided Visual Pre-training. [[ArXiv'2022](https://arxiv.org/abs/2203.05175)]
* **Ge2AE**: Hao Liu, Xinghua Jiang, Xin Li, Antai Guo, Deqiang Jiang, Bo Ren.
   - The Devil is in the Frequency: Geminated Gestalt Autoencoder for Self-Supervised Visual Pre-Training. [[ArXiv'2022](https://arxiv.org/abs/2204.08227)]
* **ConvMAE**: Peng Gao, Teli Ma, Hongsheng Li, Ziyi Lin, Jifeng Dai, Yu Qiao.
   - ConvMAE: Masked Convolution Meets Masked Autoencoders. [[NIPS'2022](https://arxiv.org/abs/2205.03892)] [[code](https://github.com/Alpha-VL/ConvMAE)]
* **GreenMIM**: Lang Huang, Shan You, Mingkai Zheng, Fei Wang, Chen Qian, Toshihiko Yamasaki.
   - Green Hierarchical Vision Transformer for Masked Image Modeling. [[NIPS'2022](https://arxiv.org/abs/2205.13515)] [[code](https://github.com/LayneH/GreenMIM)]
* **HiViT**: Xiaosong Zhang, Yunjie Tian, Wei Huang, Qixiang Ye, Qi Dai, Lingxi Xie, Qi Tian.
   - HiViT: Hierarchical Vision Transformer Meets Masked Image Modeling. [[ArXiv'2022](https://arxiv.org/abs/2205.14949)]
* **FD**: Yixuan Wei, Han Hu, Zhenda Xie, Zheng Zhang, Yue Cao, Jianmin Bao, Dong Chen, Baining Guo.
   - Contrastive Learning Rivals Masked Image Modeling in Fine-tuning via Feature Distillation. [[ArXiv'2022](https://arxiv.org/abs/2205.14141)] [[code](https://github.com/SwinTransformer/Feature-Distillation)]
* **ObjMAE**: Jiantao Wu, Shentong Mo.
   - Object-wise Masked Autoencoders for Fast Pre-training. [[ArXiv'2022](https://arxiv.org/abs/2205.14338)]
* **LoMaR**: Jun Chen, Ming Hu, Boyang Li, Mohamed Elhoseiny.
   - Efficient Self-supervised Vision Pretraining with Local Masked Reconstruction. [[ArXiv'2022](https://arxiv.org/abs/2206.00790)] [[code](https://github.com/junchen14/LoMaR)]
* **ExtreMA**: Zhirong Wu, Zihang Lai, Xiao Sun, Stephen Lin.
   - Extreme Masking for Learning Instance and Distributed Visual Representations. [[ArXiv'2022](https://arxiv.org/abs/2206.04667)]
* **BEiT.V2**: Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, Furu Wei.
   - BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers. [[ArXiv'2022](http://arxiv.org/abs/2208.06366)] [[code](https://aka.ms/beit)]
* **MILAN**: Zejiang Hou, Fei Sun, Yen-Kuang Chen, Yuan Xie, Sun-Yuan Kung.
   - MILAN: Masked Image Pretraining on Language Assisted Representation. [[ArXiv'2022](https://arxiv.org/abs/2208.06049)] [[code](https://github.com/zejiangh/milan)]
* **BEiT.V3**: Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, Furu Wei.
   - Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks. [[ArXiv'2022](https://arxiv.org/abs/2208.10442)] [[code](https://github.com/microsoft/unilm/tree/master/beit)]
* **DMAE**: Yutong Bai, Zeyu Wang, Junfei Xiao, Chen Wei, Huiyu Wang, Alan Yuille, Yuyin Zhou, Cihang Xie.
   - Masked Autoencoders Enable Efficient Knowledge Distillers. [[ArXiv'2022](https://arxiv.org/abs/2208.12256)] [[code](https://github.com/UCSC-VLAA/DMAE)]

### MIM with Constrastive Learning

* **MST**: Zhaowen Li, Zhiyang Chen, Fan Yang, Wei Li, Yousong Zhu, Chaoyang Zhao, Rui Deng, Liwei Wu, Rui Zhao, Ming Tang, Jinqiao Wang.
   - MST: Masked Self-Supervised Transformer for Visual Representation. [[NIPS'2021](https://arxiv.org/abs/2106.05656)]
* **SplitMask**: Alaaeldin El-Nouby, Gautier Izacard, Hugo Touvron, Ivan Laptev, Hervé Jegou, Edouard Grave.
   - Are Large-scale Datasets Necessary for Self-Supervised Pre-training? [[ArXiv'2021](https://arxiv.org/abs/2112.10740)]
* **MSN**: Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Michael Rabbat, Nicolas Ballas.
   - Masked Siamese Networks for Label-Efficient Learning. [[ArXiv'2022](https://arxiv.org/abs/2204.07141)] [[code](https://github.com/facebookresearch/msn)]
* **SIM**: Chenxin Tao, Xizhou Zhu, Gao Huang, Yu Qiao, Xiaogang Wang, Jifeng Dai.
   - Siamese Image Modeling for Self-Supervised Vision Representation Learning. [[ArXiv'2022](https://arxiv.org/abs/2206.01204)] [[code](https://github.com/fundamentalvision/Siamese-Image-Modeling)]
* **ConMIM**: Kun Yi, Yixiao Ge, Xiaotong Li, Shusheng Yang, Dian Li, Jianping Wu, Ying Shan, Xiaohu Qie.
   - Masked Image Modeling with Denoising Contrast. [[ArXiv'2022](https://arxiv.org/abs/2205.09616)]
* **RePre**: Luya Wang, Feng Liang, Yangguang Li, Honggang Zhang, Wanli Ouyang, Jing Shao.
   - RePre: Improving Self-Supervised Vision Transformer with Reconstructive Pre-training. [[ArXiv'2022](https://arxiv.org/abs/2201.06857)]
* **CMAE**: Zhicheng Huang, Xiaojie Jin, Chengze Lu, Qibin Hou, Ming-Ming Cheng, Dongmei Fu, Xiaohui Shen, Jiashi Feng.
   - Contrastive Masked Autoencoders are Stronger Vision Learners. [[ArXiv'2022](https://arxiv.org/abs/2207.13532)]

### MIM for Transformer and CNN

* **Context-Encoder**: Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros.
   - Context Encoders: Feature Learning by Inpainting. [[CVPR'2016](https://arxiv.org/abs/1604.07379)] [[code](https://github.com/pathak22/context-encoder)]
* **CIM**: Yuxin Fang, Li Dong, Hangbo Bao, Xinggang Wang, Furu Wei.
   - Corrupted Image Modeling for Self-Supervised Visual Pre-Training. [[ArXiv'2022](https://arxiv.org/abs/2202.03382)]
* **A2MIM**: Siyuan Li, Di Wu, Fang Wu, Zelin Zang, Kai Wang, Lei Shang, Baigui Sun, Hao Li, Stan.Z.Li.
   - Architecture-Agnostic Masked Image Modeling - From ViT back to CNN. [[ArXiv'2022](https://arxiv.org/abs/2205.13943)] [[code](https://github.com/Westlake-AI/openmixup)]
* **MFM**: Jiahao Xie, Wei Li, Xiaohang Zhan, Ziwei Liu, Yew Soon Ong, Chen Change Loy.
   - Masked Frequency Modeling for Self-Supervised Visual Pre-Training. [[ArXiv'2022](https://arxiv.org/abs/2206.07706)] [[code](https://www.mmlab-ntu.com/project/mfm/index.html)]
* **MixMIM**: Jihao Liu, Xin Huang, Yu Liu, Hongsheng Li.
   - MixMIM: Mixed and Masked Image Modeling for Efficient Visual Representation Learning. [[ArXiv'2022](https://arxiv.org/abs/2205.13137)] [[code](https://github.com/Sense-X/MixMIM)]
* **MRA**: Haohang Xu, Shuangrui Ding, Xiaopeng Zhang, Hongkai Xiong, Qi Tian.
   - Masked Autoencoders are Robust Data Augmentors. [[ArXiv'2022](https://arxiv.org/abs/2206.04846)] [[code](https://github.com/haohang96/mra)]

### MIM with Advanced Masking

* **ADIOS**: Yuge Shi, N. Siddharth, Philip H.S. Torr, Adam R. Kosiorek.
   - Adversarial Masking for Self-Supervised Learning. [[ICML'2022](https://arxiv.org/abs/2201.13100)] [[code](https://github.com/YugeTen/adios)]
* **AttMask**: Ioannis Kakogeorgiou, Spyros Gidaris, Bill Psomas, Yannis Avrithis, Andrei Bursuc, Konstantinos Karantzalos, Nikos Komodakis.
   - What to Hide from Your Students: Attention-Guided Masked Image Modeling. [[ECCV'2022](https://arxiv.org/abs/2203.12719)] [[code](https://github.com/gkakogeorgiou/attmask)]
* **UnMAE**: Xiang Li, Wenhai Wang, Lingfeng Yang, Jian Yang.
   - Uniform Masking: Enabling MAE Pre-training for Pyramid-based Vision Transformers with Locality. [[ArXiv'2022](https://arxiv.org/abs/2205.10063)] [[code](https://github.com/implus/um-mae)]
* **SemMAE**: Gang Li, Heliang Zheng, Daqing Liu, Chaoyue Wang, Bing Su, Changwen Zheng.
   - SemMAE: Semantic-Guided Masking for Learning Masked Autoencoders. [[NIPS'2022](https://arxiv.org/abs/2206.10207)]


## MIM for Downstream Tasks

### Object Detection

* **MIMDet**: Yuxin Fang, Shusheng Yang, Shijie Wang, Yixiao Ge, Ying Shan, Xinggang Wang.
   - Unleashing Vanilla Vision Transformer with Masked Image Modeling for Object Detection. [[ArXiv'2022](https://arxiv.org/abs/2204.02964)] [[code](https://github.com/hustvl/MIMDet)]

### Video Rrepresentation

* **VideoMAE**: Zhan Tong, Yibing Song, Jue Wang, Limin Wang.
   - VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. [[NIPS'2022](https://arxiv.org/abs/2203.12602)] [[code](https://github.com/MCG-NJU/VideoMAE)]
* **MAE**: Christoph Feichtenhofer, Haoqi Fan, Yanghao Li, Kaiming He.
   - Masked Autoencoders As Spatiotemporal Learners. [[NIPS'2022](https://arxiv.org/abs/2205.09113)] [[code](https://github.com/facebookresearch/SlowFast)]
* **FMNet**: Yiran Wang, Zhiyu Pan, Xingyi Li, Zhiguo Cao, Ke Xian, Jianming Zhang.
   - Less is More: Consistent Video Depth Estimation with Masked Frames Modeling. [[ACMMM'2022](https://arxiv.org/abs/2208.00380)] [[code](https://github.com/RaymondWang987/FMNet)]
* **MaskViT**: Agrim Gupta, Stephen Tian, Yunzhi Zhang, Jiajun Wu, Roberto Martín-Martín, Li Fei-Fei.
   - MaskViT: Masked Visual Pre-Training for Video Prediction. [[ArXiv'2022](https://arxiv.org/abs/2206.11894)] [[code](https://github.com/agrimgupta92/maskvit)]
* **OmniMAE**: Rohit Girdhar, Alaaeldin El-Nouby, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra.
   - OmniMAE: Single Model Masked Pretraining on Images and Videos. [[ArXiv'2022](http://arxiv.org/abs/2206.08356)] [[code](https://github.com/facebookresearch/omnivore)]
* **MILES**: Yuying Ge, Yixiao Ge, Xihui Liu, Alex Jinpeng Wang, Jianping Wu, Ying Shan, Xiaohu Qie, Ping Luo.
   - MILES: Visual BERT Pre-training with Injected Language Semantics for Video-text Retrieval. [[ArXiv'2022](https://arxiv.org/abs/2204.12408)] [[code](https://github.com/tencentarc/mcq)]
* **MAR**: Zhiwu Qing, Shiwei Zhang, Ziyuan Huang, Xiang Wang, Yuehuan Wang, Yiliang Lv, Changxin Gao, Nong Sang.
   - MAR: Masked Autoencoders for Efficient Action Recognition. [[ArXiv'2022](http://arxiv.org/abs/2207.11660)]

### Medical Image

* **MedMAE**: Lei Zhou, Huidong Liu, Joseph Bae, Junjun He, Dimitris Samaras, Prateek Prasanna.
   - Self Pre-training with Masked Autoencoders for Medical Image Analysis. [[ArXiv'2022](https://arxiv.org/abs/2203.05573)]
* **SD-MAE**: Yang Luo, Zhineng Chen, Xieping Gao.
   - Self-distillation Augmented Masked Autoencoders for Histopathological Image Classification. [[ArXiv'2022](https://arxiv.org/abs/2203.16983)]
* **GCMAE**: Hao Quan, Xingyu Li, Weixing Chen, Qun Bai, Mingchen Zou, Ruijie Yang, Tingting Zheng, Ruiqun Qi, Xinghua Gao, Xiaoyu Cui.
   - Global Contrast Masked Autoencoders Are Powerful Pathological Representation Learners. [[ArXiv'2022](https://arxiv.org/abs/2205.09048)] [[code](https://github.com/staruniversus/gcmae)]

### Face Recognition

* **FaceMAE**: Kai Wang, Bo Zhao, Xiangyu Peng, Zheng Zhu, Jiankang Deng, Xinchao Wang, Hakan Bilen, Yang You.
   - FaceMAE: Privacy-Preserving Face Recognition via Masked Autoencoders. [[ArXiv'2022](https://arxiv.org/abs/2205.11090)] [[code](https://github.com/kaiwang960112/FaceMAE)]

### Scene Text Recognition (OCR)

* **MaskOCR**: Pengyuan Lyu, Chengquan Zhang, Shanshan Liu, Meina Qiao, Yangliu Xu, Liang Wu, Kun Yao, Junyu Han, Errui Ding, Jingdong Wang.
   - MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining. [[ArXiv'2022](https://arxiv.org/abs/2206.00311)]

### Satellite Imagery

* **SatMAE**: Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David B. Lobell, Stefano Ermon.
   - SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery. [[ArXiv'2022](https://arxiv.org/abs/2207.08051)]

### 3D Point Cloud

* **PointBERT**: Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, Jiwen Lu.
   - Pre-Training 3D Point Cloud Transformers with Masked Point Modeling. [[CVPR'2022](https://arxiv.org/abs/2111.14819)] [[code](https://github.com/lulutang0608/Point-BERT)]
* **PointMAE**: Yatian Pang, Wenxiao Wang, Francis E.H. Tay, Wei Liu, Yonghong Tian, Li Yuan.
   - Masked Autoencoders for Point Cloud Self-supervised Learning. [[ECCV'2022](https://arxiv.org/abs/2203.06604)] [[code](https://github.com/Pang-Yatian/Point-MAE)]
* **MaskPoint**: Haotian Liu, Mu Cai, Yong Jae Lee.
   - Masked Discrimination for Self-Supervised Learning on Point Clouds. [[ECCV'2022](https://arxiv.org/abs/2203.11183)] [[code](https://github.com/haotian-liu/MaskPoint)]
* **MeshMAE**: Yaqian Liang, Shanshan Zhao, Baosheng Yu, Jing Zhang, Fazhi He.
   - MeshMAE: Masked Autoencoders for 3D Mesh Data Analysis. [[ECCV'2022](http://arxiv.org/abs/2207.10228)]
* **VoxelMAE**: Chen Min, Xinli Xu, Dawei Zhao, Liang Xiao, Yiming Nie, Bin Dai.
   - Voxel-MAE: Masked Autoencoders for Pre-training Large-scale Point Clouds. [[ArXiv'2022](https://arxiv.org/abs/2206.09900)]
* **Point-M2AE**: Renrui Zhang, Ziyu Guo, Peng Gao, Rongyao Fang, Bin Zhao, Dong Wang, Yu Qiao, Hongsheng Li.
   - Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training. [[NIPS'2022](https://arxiv.org/abs/2205.14401)] [[code](https://github.com/zrrskywalker/point-m2ae)]

### Reinforcement Learning

* **MLR**: Tao Yu, Zhizheng Zhang, Cuiling Lan, Yan Lu, Zhibo Chen.
   - Mask-based Latent Reconstruction for Reinforcement Learning. [[ArXiv'2022](https://arxiv.org/abs/2201.12096)]

### Audio

* **MAE-AST**: Alan Baade, Puyuan Peng, David Harwath.
   - MAE-AST: Masked Autoencoding Audio Spectrogram Transformer. [[ArXiv'2022](https://arxiv.org/abs/2203.16691)] [[code](https://github.com/AlanBaade/MAE-AST-Public)]
* **MaskSpec**: Dading Chong, Helin Wang, Peilin Zhou, Qingcheng Zeng.
   - Masked Spectrogram Prediction For Self-Supervised Audio Pre-Training. [[ArXiv'2022](https://arxiv.org/abs/2204.12768)] [[code](https://github.com/wanghelin1997/maskspec)]
* **Audio-MAE**: Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, Christoph Feichtenhofer.
   - Masked Autoencoders that Listen. [[NIPS'2022](https://arxiv.org/abs/2207.06405)] [[code](https://github.com/facebookresearch/audiomae)]


## Analysis of MIM

* Yao-Hung Hubert Tsai, Yue Wu, Ruslan Salakhutdinov, Louis-Philippe Morency.
   - Demystifying Self-Supervised Learning: An Information-Theoretical Framework. [[ICLR'2021](https://arxiv.org/abs/2006.05576)] [[code](https://github.com/yaohungt/Self_Supervised_Learning_Multiview)]
* Nikunj Saunshi, Sadhika Malladi, Sanjeev Arora.
   - A Mathematical Exploration of Why Language Models Help Solve Downstream Tasks. [[ICLR'2021](https://arxiv.org/abs/2010.03648)]
* Jason D. Lee, Qi Lei, Nikunj Saunshi, Jiacheng Zhuo.
   - Predicting What You Already Know Helps: Provable Self-Supervised Learning. [[NIPS'2021](https://arxiv.org/abs/2008.01064)]
* Shuhao Cao, Peng Xu, David A. Clifton.
   - How to Understand Masked Autoencoders. [[ArXiv'2022](https://arxiv.org/abs/2202.03670)]
* Bingbin Liu, Daniel Hsu, Pradeep Ravikumar, Andrej Risteski.
   - Masked prediction tasks: a parameter identifiability view. [[ArXiv'2022](https://arxiv.org/abs/2202.09305)]
* Zhenda Xie, Zigang Geng, Jingcheng Hu, Zheng Zhang, Han Hu, Yue Cao.
   - Revealing the Dark Secrets of Masked Image Modeling. [[ArXiv'2022](https://arxiv.org/abs/2205.13543)]
* Siyuan Li, Di Wu, Fang Wu, Zelin Zang, Kai Wang, Lei Shang, Baigui Sun, Hao Li, Stan.Z.Li.
   - Architecture-Agnostic Masked Image Modeling - From ViT back to CNN. [[ArXiv'2022](https://arxiv.org/abs/2205.13943)] [[code](https://github.com/Westlake-AI/openmixup)]
* Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Yixuan Wei, Qi Dai, Han Hu.
   - On Data Scaling in Masked Image Modeling. [[ArXiv'2022](https://arxiv.org/abs/2206.04664)]
* Jiachun Pan, Pan Zhou, Shuicheng Yan.
   - Towards Understanding Why Mask-Reconstruction Pretraining Helps in Downstream Tasks. [[ArXiv'2022](https://arxiv.org/abs/2206.03826)]
* Gokul Karthik Kumar, Sahal Shaji Mullappilly, Abhishek Singh Gehlot.
   - An Empirical Study Of Self-supervised Learning Approaches For Object Detection With Transformers. [[ArXiv'2022](https://arxiv.org/abs/2205.05543)] [[code](https://github.com/gokulkarthik/deformable-detr)]
* Xiangwen Kong, Xiangyu Zhang.
   - Understanding Masked Image Modeling via Learning Occlusion Invariant Feature. [[ArXiv'2022](http://arxiv.org/abs/2208.04164)]
* Qi Zhang, Yifei Wang, Yisen Wang.
   - How Mask Matters: Towards Theoretical Understandings of Masked Autoencoders. [[NIP'2022](https://openreview.net/forum?id=WOppMAJtvhv)]


## Contribution

Feel free to send [pull requests](https://github.com/Westlake-AI/openmixup/pulls) to add more links with the following Markdown format. Current contributors include: Siyuan Li ([@Lupin1998](https://github.com/Lupin1998)).

```markdown
* **Abbreviation**: Author List.
  - Paper Name. [[Conference'Year](link)] [[code](link)]
```


## Related Project

### Paper List of Masked Image Modeling

- [awesome-MIM](https://github.com/ucasligang/awesome-MIM): Reading list for research topics in Masked Image Modeling.
- [Awesome-MIM](https://github.com/Lupin1998/Awesome-MIM): Awesome List of Masked Image Modeling (MIM) Papers for Self-supervised Visual Representation Learning.
- [awesome-self-supervised-learning](https://github.com/jason718/awesome-self-supervised-learning): A curated list of awesome self-supervised methods.

### Porject of Self-supervised Learning

- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [solo-learn](https://github.com/vturrisi/solo-learn): A library of self-supervised methods for visual representation learning powered by Pytorch Lightning.
- [unilm](https://github.com/microsoft/unilm): Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities.
- [VISSL](https://github.com/facebookresearch/vissl): FAIR's library of extensible, modular and scalable components for SOTA Self-Supervised Learning with images. 
