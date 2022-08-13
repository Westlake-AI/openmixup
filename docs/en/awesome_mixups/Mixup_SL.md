# Awesome Mixup Methods for Supervised Learning

 [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![GitHub stars](https://img.shields.io/github/stars/Westlake-AI/openmixup?color=green) ![visitors](https://visitor-badge.glitch.me/badge?page_id=Westlake-AI/openmixup)

**We summarize mixup methods proposed for supervised visual representation learning from two aspects: *sample mixup policy* and *label mixup policy*.**
We are working on a survey of mixup methods. The list of awesome mixup methods is summarized in chronological order and is on updating.

## Sample Mixup Methods

### Pre-defined Policies

* **MixUp**: Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz.
   - mixup: Beyond Empirical Risk Minimization. [[ICLR'2018](https://arxiv.org/abs/1710.09412)] [[code](https://github.com/facebookresearch/mixup-cifar10)]
* **AdaMixup**: Hongyu Guo, Yongyi Mao, Richong Zhang.
   - MixUp as Locally Linear Out-Of-Manifold Regularization. [[AAAI'2019](https://arxiv.org/abs/1809.02499)]
* **CutMix**: Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo.
   - CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. [[ICCV'2019](https://arxiv.org/abs/1905.04899)] [[code](https://github.com/clovaai/CutMix-PyTorch)]
* **ManifoldMix**: Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas, David Lopez-Paz, Yoshua Bengio.
   - Manifold Mixup: Better Representations by Interpolating Hidden States. [[ICML'2019](https://arxiv.org/abs/1806.05236)] [[code](https://github.com/vikasverma1077/manifold_mixup)]
* **FMix**: Ethan Harris, Antonia Marcu, Matthew Painter, Mahesan Niranjan, Adam Prügel-Bennett, Jonathon Hare.
   - FMix: Enhancing Mixed Sample Data Augmentation. [[Arixv'2020](https://arxiv.org/abs/2002.12047)] [[code](https://github.com/ecs-vlc/FMix)]
* **SmoothMix**: Jin-Ha Lee, Muhammad Zaigham Zaheer, Marcella Astrid, Seung-Ik Lee.
   - SmoothMix: a Simple Yet Effective Data Augmentation to Train Robust Classifiers. [[CVPRW'2020](https://arxiv.org/abs/2002.12047)] [[code](https://github.com/Westlake-AI/openmixup)]
* **PatchUp**: Mojtaba Faramarzi, Mohammad Amini, Akilesh Badrinaaraayanan, Vikas Verma, Sarath Chandar.
   - PatchUp: A Regularization Technique for Convolutional Neural Networks. [[Arxiv'2020](https://arxiv.org/abs/2006.07794)] [[code](https://github.com/chandar-lab/PatchUp)]
* **GridMixup**: Kyungjune Baek, Duhyeon Bang, Hyunjung Shim.
   - GridMix: Strong regularization through local context mapping. [[Pattern Recognition'2021](https://www.sciencedirect.com/science/article/pii/S0031320320303976)] [[code](https://github.com/IlyaDobrynin/GridMixup)]
* **ResizeMix**: Jie Qin, Jiemin Fang, Qian Zhang, Wenyu Liu, Xingang Wang, Xinggang Wang.
   - ResizeMix: Mixing Data with Preserved Object Information and True Labels. [[Arixv'2020](https://arxiv.org/abs/2012.11101)] [[code](https://github.com/Westlake-AI/openmixup)]
* **FocusMix**: Jiyeon Kim, Ik-Hee Shin, Jong-Ryul, Lee, Yong-Ju Lee.
   - Where to Cut and Paste: Data Regularization with Selective Features. [[ICTC'2020](https://ieeexplore.ieee.org/abstract/document/9289404)]
* **AugMix**: Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, Balaji Lakshminarayanan.
   - AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty. [[ICLR'2020](https://arxiv.org/abs/1912.02781)] [[code](https://github.com/google-research/augmix)]
* **DJMix**: Ryuichiro Hataya, Hideki Nakayama.
   - DJMix: Unsupervised Task-agnostic Augmentation for Improving Robustness. [[Arxiv'2021](https://openreview.net/pdf?id=0n3BaVlNsHI)]
* **PixMix**: Dan Hendrycks, Andy Zou, Mantas Mazeika, Leonard Tang, Bo Li, Dawn Song, Jacob Steinhardt.
   - PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures. [[Arxiv'2021](https://arxiv.org/abs/2112.05135)] [[code](https://github.com/andyzoujm/pixmix)]
* **StyleMix**: Minui Hong, Jinwoo Choi, Gunhee Kim.
   - StyleMix: Separating Content and Style for Enhanced Data Augmentation. [[CVPR'2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_StyleMix_Separating_Content_and_Style_for_Enhanced_Data_Augmentation_CVPR_2021_paper.pdf)] [[code](https://github.com/alsdml/StyleMix)]
* **MixStyle**: Kaiyang Zhou, Yongxin Yang, Yu Qiao, Tao Xiang.
   - Domain Generalization with MixStyle. [[ICLR'2021](https://openreview.net/forum?id=6xHJ37MVxxp)] [[code](https://github.com/KaiyangZhou/mixstyle-release)]
* **MoEx**: Boyi Li, Felix Wu, Ser-Nam Lim, Serge Belongie, Kilian Q. Weinberger.
   - On Feature Normalization and Data Augmentation. [[CVPR'2021](https://arxiv.org/abs/2002.11102)] [[code](https://github.com/Boyiliee/MoEx)]
* **LocalMix**: Raphael Baena, Lucas Drumetz, Vincent Gripon.
   - Preventing Manifold Intrusion with Locality: Local Mixup. [[AISTATS'2021](https://arxiv.org/abs/2201.04368)]

### Saliency-guided Policies

* **SaliencyMix**: A F M Shahab Uddin and Mst. Sirazam Monira and Wheemyung Shin and TaeChoong Chung and Sung-Ho Bae
   - SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization. [[ICLR'2021](https://arxiv.org/abs/2006.01791)] [[code](https://github.com/SaliencyMix/SaliencyMix)]
* **AttentiveMix**: Devesh Walawalkar, Zhiqiang Shen, Zechun Liu, Marios Savvides.
   - Attentive CutMix: An Enhanced Data Augmentation Approach for Deep Learning Based Image Classification. [[ICASSP'2020](https://arxiv.org/abs/2003.13048)] [[code](https://github.com/xden2331/attentive_cutmix)]
* **SnapMix**: Shaoli Huang, Xinchao Wang, Dacheng Tao.
   - SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data. [[AAAI'2021](https://arxiv.org/abs/2012.04846)] [[code](https://github.com/Shaoli-Huang/SnapMix)]
* **AttributeMix**: Hao Li, Xiaopeng Zhang, Hongkai Xiong, Qi Tian.
   - Attribute Mix: Semantic Data Augmentation for Fine Grained Recognition. [[Arxiv'2020](https://arxiv.org/abs/2004.02684)]
* **AutoMix**: Jianchao Zhu, Liangliang Shi, Junchi Yan, Hongyuan Zha.
   - AutoMix: Mixup Networks for Sample Interpolation via Cooperative Barycenter Learning. [[ECCV'2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550630.pdf)]
* **PuzzleMix**: Jang-Hyun Kim, Wonho Choo, Hyun Oh Song.
   - Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup. [[ICML'2020](https://arxiv.org/abs/2009.06962)] [[code](https://github.com/snu-mllab/PuzzleMix)]
* **CoMixup**: Jang-Hyun Kim, Wonho Choo, Hosan Jeong, Hyun Oh Song.
   - Co-Mixup: Saliency Guided Joint Mixup with Supermodular Diversity. [[ICLR'2021](https://arxiv.org/abs/2102.03065)] [[code](https://github.com/snu-mllab/Co-Mixup)]
* **SuperMix**: Ali Dabouei, Sobhan Soleymani, Fariborz Taherkhani, Nasser M. Nasrabadi.
   - SuperMix: Supervising the Mixing Data Augmentation. [[CVPR'2021](https://arxiv.org/abs/2003.05034)] [[code](https://github.com/alldbi/SuperMix)]
* **PatchMix**: Paola Cascante-Bonilla, Arshdeep Sekhon, Yanjun Qi, Vicente Ordonez.
   - Evolving Image Compositions for Feature Representation Learning. [[BMVC'2021](https://arxiv.org/pdf/2106.09011.pdf)]
* **StackMix**: John Chen, Samarth Sinha, Anastasios Kyrillidis.
   - StackMix: A complementary Mix algorithm. [[Arxiv'2021](https://arxiv.org/abs/2011.12618)]
* **AlignMix**: Shashanka Venkataramanan, Ewa Kijak, Laurent Amsaleg, Yannis Avrithis.
   - AlignMix: Improving representation by interpolating aligned features. [[CVPR'2022](https://arxiv.org/abs/2103.15375)] [[code](https://github.com/shashankvkt/AlignMixup_CVPR22)]
* **AutoMix**: Zicheng Liu, Siyuan Li, Di Wu, Zihan Liu, Zhiyuan Chen, Lirong Wu, Stan Z. Li.
   - AutoMix: Unveiling the Power of Mixup for Stronger Classifiers. [[ECCV'2022](https://arxiv.org/abs/2103.13027)] [[code](https://github.com/Westlake-AI/openmixup)]
* **SAMix**: Siyuan Li, Zicheng Liu, Di Wu, Zihan Liu, Stan Z. Li.
   - Boosting Discriminative Visual Representation Learning with Scenario-Agnostic Mixup. [[Arxiv'2021](https://arxiv.org/abs/2111.15454)] [[code](https://github.com/Westlake-AI/openmixup)]
* **ScoreMix**: Thomas Stegmüller, Behzad Bozorgtabar, Antoine Spahr, Jean-Philippe Thiran.
   - ScoreNet: Learning Non-Uniform Attention and Augmentation for Transformer-Based Histopathological Image Classification. [[Arxiv'2022](https://arxiv.org/abs/2202.07570)]
* **RecursiveMix**: Lingfeng Yang, Xiang Li, Borui Zhao, Renjie Song, Jian Yang.
   - RecursiveMix: Mixed Learning with History. [[Arxiv'2022](https://arxiv.org/abs/2203.06844)] [[code](https://github.com/implus/RecursiveMix-pytorch)]

## Label Mixup Methods

* **MixUp**: Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz.
   - mixup: Beyond Empirical Risk Minimization. [[ICLR'2018](https://arxiv.org/abs/1710.09412)] [[code](https://github.com/facebookresearch/mixup-cifar10)]
* **CutMix**: Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo.
   - CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. [[ICCV'2019](https://arxiv.org/abs/1905.04899)] [[code](https://github.com/clovaai/CutMix-PyTorch)]
* **MetaMixup**: Zhijun Mai, Guosheng Hu, Dexiong Chen, Fumin Shen, Heng Tao Shen.
   - Metamixup: Learning adaptive interpolation policy of mixup with metalearning. [[TNNLS'2021](https://arxiv.org/abs/1908.10059)]
* **mWH**: Hao Yu, Huanyu Wang, Jianxin Wu.
   - Mixup Without Hesitation. [[Pattern Recognition'2022](https://arxiv.org/abs/2101.04342)] [[code](https://github.com/yuhao318/mwh)]
* **CAMixup**: Yeming Wen, Ghassen Jerfel, Rafael Muller, Michael W. Dusenberry, Jasper Snoek, Balaji Lakshminarayanan, Dustin Tran.
   - Combining Ensembles and Data Augmentation can Harm your Calibration. [[ICLR'2021](https://arxiv.org/abs/2010.09875)] [[code](https://github.com/google/edward2/tree/main/experimental/marginalization_mixup)]
* **Saliency Grafting**: Joonhyung Park, June Yong Yang, Jinwoo Shin, Sung Ju Hwang, Eunho Yang.
   - Saliency Grafting: Innocuous Attribution-Guided Mixup with Calibrated Label Mixing. [[AAAI'2022](https://arxiv.org/abs/2112.08796)]
* **TransMix**: Jie-Neng Chen, Shuyang Sun, Ju He, Philip Torr, Alan Yuille, Song Bai.
   - TransMix: Attend to Mix for Vision Transformers. [[CVPR'2022](https://arxiv.org/abs/2111.09833)] [[code](https://github.com/Beckschen/TransMix)]
* **DecoupleMix**: Zicheng Liu, Siyuan Li, Ge Wang, Cheng Tan, Lirong Wu, Stan Z. Li.
   - Decoupled Mixup for Data-efficient Learning. [[Arxiv'2022](https://arxiv.org/abs/2203.10761),] [[code](https://github.com/Westlake-AI/openmixup)]
* **TokenMix**: Jihao Liu, Boxiao Liu, Hang Zhou, Hongsheng Li, Yu Liu.
   - TokenMix: Rethinking Image Mixing for Data Augmentation in Vision Transformers. [[ECCV'2022](https://arxiv.org/abs/2207.08409),] [[code](https://github.com/Sense-X/TokenMix)]

## Contribution

Feel free to send [pull requests](https://github.com/Westlake-AI/openmixup/pulls) to add more links! Current contributors include: Siyuan Li ([@Lupin1998](https://github.com/Lupin1998)) and Zicheng Liu ([@pone7](https://github.com/pone7)).
