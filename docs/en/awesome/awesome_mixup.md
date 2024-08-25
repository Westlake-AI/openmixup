# Awesome Mixup Methods for Supervised Learning

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![GitHub stars](https://img.shields.io/github/stars/Westlake-AI/openmixup?color=blue) ![GitHub forks](https://img.shields.io/github/forks/Westlake-AI/openmixup?color=yellow&label=Fork)

**We summarize fundamental mixup methods proposed for supervised visual representation learning from two aspects: *sample mixup policy* and *label mixup policy*. Then, we summarize mixup techniques used in downstream tasks.**
The list of awesome mixup methods is summarized in chronological order and is on updating. And we will add more papers according to [Awesome-Mix](https://github.com/ChengtaiCao/Awesome-Mix).

* To find related papers and their relationships, check out [Connected Papers](https://www.connectedpapers.com/), which visualizes the academic field in a graph representation.
* To export BibTeX citations of papers, check out [ArXiv](https://arxiv.org/) or [Semantic Scholar](https://www.semanticscholar.org/) of the paper for professional reference formats.

## Table of Contents

- [Awesome-Mixup](#awesome-mixup)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Fundermental Methods](#fundermental-methods)
    - [Sample Mixup Methods](#sample-mixup-methods)
      - [**Pre-defined Policies**](#pre-defined-policies)
      - [**Adaptive Policies**](#adaptive-policies)
    - [Label Mixup Methods](#label-mixup-methods)
  - [Mixup for Self-supervised Learning](#mixup-for-self-supervised-learning)
  - [Mixup for Semi-supervised Learning](#mixup-for-semi-supervised-learning)
  - [Mixup for Regression](#mixup-for-regression)
  - [Mixup for Robustness](#mixup-for-robustness)
  - [Mixup for Multi-modality](#mixup-for-multi-modality)
  - [Analysis of Mixup](#analysis-of-mixup)
  - [Natural Language Processing](#natural-language-processing)
  - [Graph Representation Learning](#graph-representation-learning)
  - [Survey](#survey)
  - [Benchmark](#benchmark)
  - [Contribution](#contribution)
  - [License](#license)
  - [Acknowledgement](#acknowledgement)
  - [Related Project](#related-project)


## Fundermental Methods

### Sample Mixup Methods

#### **Pre-defined Policies**

* **mixup: Beyond Empirical Risk Minimization**<br>
*Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz*<br>
ICLR'2018 [[Paper](https://arxiv.org/abs/1710.09412)]
[[Code](https://github.com/facebookresearch/mixup-cifar10)]
   <details close>
   <summary>MixUp Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204561478-80b77110-21a4-480f-b369-d2f0656b5382.png" /></p>
   </details>

* **Between-class Learning for Image Classification**<br>
*Yuji Tokozume, Yoshitaka Ushiku, Tatsuya Harada*<br>
CVPR'2018 [[Paper](https://arxiv.org/abs/1711.10284)]
[[Code](https://github.com/mil-tokyo/bc_learning_image)]
   <details close>
   <summary>BC Framework</summary>
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/204563476-15e638ad-6c25-4ab6-9bb6-4c5be74c625f.png" /></p>
   </details>

* **MixUp as Locally Linear Out-Of-Manifold Regularization**<br>
*Hongyu Guo, Yongyi Mao, Richong Zhang*<br>
AAAI'2019 [[Paper](https://arxiv.org/abs/1809.02499)]
   <details close>
   <summary>AdaMixup Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204563766-4a49b4d9-fb1e-46d7-8443-bff37a527ee1.png" /></p>
   </details>

* **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features**<br>
*Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo*<br>
ICCV'2019 [[Paper](https://arxiv.org/abs/1905.04899)]
[[Code](https://github.com/clovaai/CutMix-PyTorch)]
   <details close>
   <summary>CutMix Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204564166-49707535-43f9-4d15-af89-d1a5a302db24.png" /></p>
   </details>

* **Manifold Mixup: Better Representations by Interpolating Hidden States**<br>
*Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas, David Lopez-Paz, Yoshua Bengio*<br>
ICML'2019 [[Paper](https://arxiv.org/abs/1806.05236)]
[[Code](https://github.com/vikasverma1077/manifold_mixup)]
   <details close>
   <summary>ManifoldMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204565193-c5416185-ed98-4b86-bc7c-f5b6cc2f839b.png" /></p>
   </details>

* **Improved Mixed-Example Data Augmentation**<br>
*Cecilia Summers, Michael J. Dinneen*<br>
WACV'2019 [[Paper](https://arxiv.org/abs/1805.11272)]
[[Code](https://github.com/ceciliaresearch/MixedExample)]
   <details close>
   <summary>MixedExamples Framework</summary>
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/232355479-8a53714f-5a6f-4c8e-b808-cc73f6d0f669.png" /></p>
   </details>

* **FMix: Enhancing Mixed Sample Data Augmentation**<br>
*Ethan Harris, Antonia Marcu, Matthew Painter, Mahesan Niranjan, Adam Prügel-Bennett, Jonathon Hare*<br>
Arixv'2020 [[Paper](https://arxiv.org/abs/2002.12047)]
[[Code](https://github.com/ecs-vlc/FMix)]
   <details close>
   <summary>FMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204564909-8d20a405-141d-4fe6-ae72-581fc24222f8.png" /></p>
   </details>

* **SmoothMix: a Simple Yet Effective Data Augmentation to Train Robust Classifiers**<br>
*Jin-Ha Lee, Muhammad Zaigham Zaheer, Marcella Astrid, Seung-Ik Lee*<br>
CVPRW'2020 [[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/html/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.html)]
[[Code](https://github.com/Westlake-AI/openmixup)]
   <details close>
   <summary>SmoothMix Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204565814-fd528402-2a57-482b-b608-1ee3096984b0.png" /></p>
   </details>

* **PatchUp: A Regularization Technique for Convolutional Neural Networks**<br>
*Mojtaba Faramarzi, Mohammad Amini, Akilesh Badrinaaraayanan, Vikas Verma, Sarath Chandar*<br>
Arxiv'2020 [[Paper](https://arxiv.org/abs/2006.07794)]
[[Code](https://github.com/chandar-lab/PatchUp)]
   <details close>
   <summary>PatchUp Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204566319-f2c67e8a-c10b-4ede-a57f-8c6ddf85e013.png" /></p>
   </details>

* **GridMix: Strong regularization through local context mapping**<br>
*Kyungjune Baek, Duhyeon Bang, Hyunjung Shim*<br>
Pattern Recognition'2021 [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320320303976)]
[[Code](https://github.com/IlyaDobrynin/GridMixup)]
   <details close>
   <summary>GridMixup Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204566566-b345e409-da4a-4f3b-b1bd-43e7c767d26a.png" /></p>
   </details>

* **ResizeMix: Mixing Data with Preserved Object Information and True Labels**<br>
*Jie Qin, Jiemin Fang, Qian Zhang, Wenyu Liu, Xingang Wang, Xinggang Wang*<br>
Arixv'2020 [[Paper](https://arxiv.org/abs/2012.11101)]
[[Code](https://github.com/Westlake-AI/openmixup)]
   <details close>
   <summary>ResizeMix Framework</summary>
   <p align="center"><img width="55%" src="https://user-images.githubusercontent.com/44519745/204566840-69782b04-4645-41b3-a6eb-428977c63881.png" /></p>
   </details>

* **Where to Cut and Paste: Data Regularization with Selective Features**<br>
*Jiyeon Kim, Ik-Hee Shin, Jong-Ryul, Lee, Yong-Ju Lee*<br>
ICTC'2020 [[Paper](https://ieeexplore.ieee.org/abstract/document/9289404)]
[[Code](https://github.com/google-research/augmix)]
   <details close>
   <summary>FocusMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204567137-f71b0437-9267-4f99-b7dc-911ffa4f8b73.png" /></p>
   </details>

* **AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty**<br>
*Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, Balaji Lakshminarayanan*<br>
ICLR'2020 [[Paper](https://arxiv.org/abs/1912.02781)]
[[Code](https://github.com/google-research/augmix)]
   <details close>
   <summary>AugMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204567137-f71b0437-9267-4f99-b7dc-911ffa4f8b73.png" /></p>
   </details>

* **DJMix: Unsupervised Task-agnostic Augmentation for Improving Robustness**<br>
*Ryuichiro Hataya, Hideki Nakayama*<br>
Arxiv'2021 [[Paper](https://openreview.net/pdf?id=0n3BaVlNsHI)]
   <details close>
   <summary>DJMix Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204567516-3d058e8c-4232-4af4-b7a8-efa3a1298a1b.png" /></p>
   </details>

* **PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures**<br>
*Dan Hendrycks, Andy Zou, Mantas Mazeika, Leonard Tang, Bo Li, Dawn Song, Jacob Steinhardt*<br>
Arxiv'2021 [[Paper](https://arxiv.org/abs/2112.05135)]
[[Code](https://github.com/andyzoujm/pixmix)]
   <details close>
   <summary>PixMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204567828-b434c118-0be1-475d-a0f3-9834e39b4507.png" /></p>
   </details>

* **StyleMix: Separating Content and Style for Enhanced Data Augmentation**<br>
*Minui Hong, Jinwoo Choi, Gunhee Kim*<br>
CVPR'2021 [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_StyleMix_Separating_Content_and_Style_for_Enhanced_Data_Augmentation_CVPR_2021_paper.pdf)]
[[Code](https://github.com/alsdml/StyleMix)]
   <details close>
   <summary>StyleMix Framework</summary>
   <p align="center"><img width="65%" src="https://user-images.githubusercontent.com/44519745/204567999-6d0263c2-111c-4335-92b5-66486bb0fea0.png" /></p>
   </details>

* **Domain Generalization with MixStyle**<br>
*Kaiyang Zhou, Yongxin Yang, Yu Qiao, Tao Xiang*<br>
ICLR'2021 [[Paper](https://openreview.net/forum?id=6xHJ37MVxxp)]
[[Code](https://github.com/KaiyangZhou/mixstyle-release)]
   <details close>
   <summary>MixStyle Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204568994-eb45528e-e83b-4ac5-bed9-642da987ec89.png" /></p>
   </details>

* **On Feature Normalization and Data Augmentation**<br>
*Boyi Li, Felix Wu, Ser-Nam Lim, Serge Belongie, Kilian Q. Weinberger*<br>
CVPR'2021 [[Paper](https://arxiv.org/abs/2002.11102)]
[[Code](https://github.com/Boyiliee/MoEx)]
   <details close>
   <summary>MoEx Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204569196-3641255a-56f3-407a-986a-209b5c7eeff6.png" /></p>
   </details>

* **Guided Interpolation for Adversarial Training**<br>
*Chen Chen, Jingfeng Zhang, Xilie Xu, Tianlei Hu, Gang Niu, Gang Chen, Masashi Sugiyama*<br>
ArXiv'2021 [[Paper](https://arxiv.org/abs/2102.07327)]
   <details close>
   <summary>GIF Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/230494660-e99cf522-8fd7-4aa4-a4e4-d0385f82d032.png" /></p>
   </details>

* **Observations on K-image Expansion of Image-Mixing Augmentation for Classification**<br>
*Joonhyun Jeong, Sungmin Cha, Youngjoon Yoo, Sangdoo Yun, Taesup Moon, Jongwon Choi*<br>
IEEE Access'2021 [[Paper](https://arxiv.org/abs/2110.04248)]
[[Code](https://github.com/yjyoo3312/DCutMix-PyTorch)]
   <details close>
   <summary>DCutMix Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/230492924-e9008de1-913f-44f5-ac2e-3d21f07d1b7f.png" /></p>
   </details>

* **Noisy Feature Mixup**<br>
*Soon Hoe Lim, N. Benjamin Erichson, Francisco Utrera, Winnie Xu, Michael W. Mahoney*<br>
ICLR'2022 [[Paper](https://arxiv.org/abs/2110.02180)]
[[Code](https://github.com/erichson/NFM)]
   <details close>
   <summary>NFM Framework</summary>
   <p align="center"><img width="45%" src="https://user-images.githubusercontent.com/44519745/204569704-7de07797-b607-4750-9a46-c4387a539ac0.png" /></p>
   </details>

* **Preventing Manifold Intrusion with Locality: Local Mixup**<br>
*Raphael Baena, Lucas Drumetz, Vincent Gripon*<br>
EUSIPCO'2022 [[Paper](https://arxiv.org/abs/2201.04368)]
[[Code](https://github.com/raphael-baena/Local-Mixup)]
   <details close>
   <summary>LocalMix Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204570088-fdf2e115-caee-4808-bf2b-709ac27f2251.png" /></p>
   </details>

* **RandomMix: A mixed sample data augmentation method with multiple mixed modes**<br>
*Xiaoliang Liu, Furao Shen, Jian Zhao, Changhai Nie*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2205.08728)]
   <details close>
   <summary>RandomMix Framework</summary>
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/204570320-2785207c-aced-4d94-a33e-98acfcbbaa3f.png" /></p>
   </details>

* **SuperpixelGridCut, SuperpixelGridMean and SuperpixelGridMix Data Augmentation**<br>
*Karim Hammoudi, Adnane Cabani, Bouthaina Slika, Halim Benhabiles, Fadi Dornaika, Mahmoud Melkemi*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2204.08458)]
[[Code](https://github.com/hammoudiproject/SuperpixelGridMasks)]
   <details close>
   <summary>SuperpixelGridCut Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204570617-090cc5a7-508c-49e7-9e5f-df7ec018f540.png" /></p>
   </details>

* **AugRmixAT: A Data Processing and Training Method for Improving Multiple Robustness and Generalization Performance**<br>
*Xiaoliang Liu, Furao Shen, Jian Zhao, Changhai Nie*<br>
ICME'2022 [[Paper](https://arxiv.org/abs/2207.10290)]
   <details close>
   <summary>AugRmixAT Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204571167-748828be-a1f6-46ac-bc13-37cccfa72515.png" /></p>
   </details>

* **A Unified Analysis of Mixed Sample Data Augmentation: A Loss Function Perspective**<br>
*Chanwoo Park, Sangdoo Yun, Sanghyuk Chun*<br>
NIPS'2022 [[Paper](https://arxiv.org/abs/2208.09913)]
[[Code](https://github.com/naver-ai/hmix-gmix)]
   <details close>
   <summary>MSDA Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204571363-f0b1a960-54f9-4462-855d-59e90f284cfe.png" /></p>
   </details>

* **RegMixup: Mixup as a Regularizer Can Surprisingly Improve Accuracy and Out Distribution Robustness**<br>
*Francesco Pinto, Harry Yang, Ser-Nam Lim, Philip H.S. Torr, Puneet K. Dokania*<br>
NIPS'2022 [[Paper](https://arxiv.org/abs/2206.14502)]
[[Code](https://github.com/FrancescoPinto/RegMixup)]
   <details close>
   <summary>RegMixup Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204571630-e8407bd7-ca27-44de-baca-5d88ca2004a6.png" /></p>
   </details>

* **ContextMix: A context-aware data augmentation method for industrial visual inspection systems**<br>
*Hyungmin Kim, Donghun Kim, Pyunghwan Ahn, Sungho Suh, Hansang Cho, Junmo Kim*<br>
EAAI'2024 [[Paper](https://arxiv.org/abs/2401.10050)]
   <details close>
   <summary>ConvtextMix Framework</summary>
   <p align="center"><img width="80%" src="https://github.com/Westlake-AI/Awesome-Mixup/assets/44519745/a2b50a9e-ca22-4c1f-9ba9-a3863b84a71a" /></p>
   </details>

<p align="right">(<a href="#top">back to top</a>)</p>

#### **Adaptive Policies**

* **SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization**<br>
*A F M Shahab Uddin and Mst. Sirazam Monira and Wheemyung Shin and TaeChoong Chung and Sung-Ho Bae*<br>
ICLR'2021 [[Paper](https://arxiv.org/abs/2006.01791)]
[[Code](https://github.com/SaliencyMix/SaliencyMix)]
   <details close>
   <summary>SaliencyMix Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204571915-624d3e5e-7678-4ba3-a08c-09ca2741bf72.png" /></p>
   </details>

* **Attentive CutMix: An Enhanced Data Augmentation Approach for Deep Learning Based Image Classification**<br>
*Devesh Walawalkar, Zhiqiang Shen, Zechun Liu, Marios Savvides*<br>
ICASSP'2020 [[Paper](https://arxiv.org/abs/2003.13048)]
[[Code](https://github.com/xden2331/attentive_cutmix)]
   <details close>
   <summary>AttentiveMix Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204572493-a8c816c9-2be5-43b6-bf35-580cfab8716f.png" /></p>
   </details>

* **SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data**<br>
*Shaoli Huang, Xinchao Wang, Dacheng Tao*<br>
AAAI'2021 [[Paper](https://arxiv.org/abs/2012.04846)]
[[Code](https://github.com/Shaoli-Huang/SnapMix)]
   <details close>
   <summary>SnapMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204572296-2035c6b4-c477-4484-a8e6-7ad9ad415045.png" /></p>
   </details>

* **Attribute Mix: Semantic Data Augmentation for Fine Grained Recognition**<br>
*Hao Li, Xiaopeng Zhang, Hongkai Xiong, Qi Tian*<br>
VCIP'2020 [[Paper](https://arxiv.org/abs/2004.02684)]
   <details close>
   <summary>AttributeMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204573220-13a3b90e-73f8-4277-a997-67dddb15dd1c.png" /></p>
   </details>

* **On Adversarial Mixup Resynthesis**<br>
*Christopher Beckham, Sina Honari, Vikas Verma, Alex Lamb, Farnoosh Ghadiri, R Devon Hjelm, Yoshua Bengio, Christopher Pal*<br>
NIPS'2019 [[Paper](https://arxiv.org/abs/1903.02709)]
[[Code](https://github.com/christopher-beckham/amr)]
   <details close>
   <summary>AMR Framework</summary>
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/232315897-2a5fb2c5-d0ce-4c01-b6cd-21ec54bd9e49.png" /></p>
   </details>

* **Patch-level Neighborhood Interpolation: A General and Effective Graph-based Regularization Strategy**<br>
*Ke Sun, Bing Yu, Zhouchen Lin, Zhanxing Zhu*<br>
ArXiv'2019 [[Paper](https://arxiv.org/abs/1911.09307)]
   <details close>
   <summary>Pani VAT Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204572993-8b3fa627-8c36-4763-a2a6-c9a90c5f0fc2.png" /></p>
   </details>

* **AutoMix: Mixup Networks for Sample Interpolation via Cooperative Barycenter Learning**<br>
*Jianchao Zhu, Liangliang Shi, Junchi Yan, Hongyuan Zha*<br>
ECCV'2020 [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550630.pdf)]
   <details close>
   <summary>AutoMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204572771-e09246ca-b88b-4755-8d8a-f99053244610.png" /></p>
   </details>

* **PuzzleMix: Exploiting Saliency and Local Statistics for Optimal Mixup**<br>
*Jang-Hyun Kim, Wonho Choo, Hyun Oh Song*<br>
ICML'2020 [[Paper](https://arxiv.org/abs/2009.06962)]
[[Code](https://github.com/snu-mllab/PuzzleMix)]
   <details close>
   <summary>PuzzleMix Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204573527-75f28e86-9b0e-4b14-bd21-ef89de52dd5f.png" /></p>
   </details>

* **Co-Mixup: Saliency Guided Joint Mixup with Supermodular Diversity**<br>
*Jang-Hyun Kim, Wonho Choo, Hosan Jeong, Hyun Oh Song*<br>
ICLR'2021 [[Paper](https://arxiv.org/abs/2102.03065)]
[[Code](https://github.com/snu-mllab/Co-Mixup)]
   <details close>
   <summary>Co-Mixup Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204573653-68ce31e8-fa01-4cf8-9493-c2311fd99e13.png" /></p>
   </details>

* **SuperMix: Supervising the Mixing Data Augmentation**<br>
*Ali Dabouei, Sobhan Soleymani, Fariborz Taherkhani, Nasser M. Nasrabadi*<br>
CVPR'2021 [[Paper](https://arxiv.org/abs/2003.05034)]
[[Code](https://github.com/alldbi/SuperMix)]
   <details close>
   <summary>SuperMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204573912-47ae05f3-8d78-4ef7-b3a9-15d54bffa20a.png" /></p>
   </details>

* **Evolving Image Compositions for Feature Representation Learning**<br>
*Paola Cascante-Bonilla, Arshdeep Sekhon, Yanjun Qi, Vicente Ordonez*<br>
BMVC'2021 [[Paper](https://arxiv.org/abs/2106.09011)]
   <details close>
   <summary>PatchMix Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204574267-8e53783d-ea54-4399-8c32-86c8ac2520bd.png" /></p>
   </details>

* **StackMix: A complementary Mix algorithm**<br>
*John Chen, Samarth Sinha, Anastasios Kyrillidis*<br>
UAI'2022 [[Paper](https://arxiv.org/abs/2011.12618)]
   <details close>
   <summary>StackMix Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204574552-fb5e9865-86b0-4d51-977c-82a840d81a18.png" /></p>
   </details>

* **SalfMix: A Novel Single Image-Based Data Augmentation Technique Using a Saliency Map**<br>
*Jaehyeop Choi, Chaehyeon Lee, Donggyu Lee, Heechul Jung*<br>
Sensor'2021 [[Paper](https://pdfs.semanticscholar.org/1db9/c80edeed50858783c69237aeba764750e8b7.pdf?_ga=2.182064935.1813772674.1674154381-1810295069.1625160008)]
   <details close>
   <summary>SalfMix Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/213535188-5255eb4c-83c8-40ca-86b9-44459b84d9a8.png" /></p>
   </details>

* **k-Mixup Regularization for Deep Learning via Optimal Transport**<br>
*Kristjan Greenewald, Anming Gu, Mikhail Yurochkin, Justin Solomon, Edward Chien*<br>
ArXiv'2021 [[Paper](https://arxiv.org/abs/2106.02933)]
   <details close>
   <summary>k-Mixup Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204569488-d4862400-3304-488d-ad24-06eff4e0c0b2.png" /></p>
   </details>

* **AlignMix: Improving representation by interpolating aligned features**<br>
*Shashanka Venkataramanan, Ewa Kijak, Laurent Amsaleg, Yannis Avrithis*<br>
CVPR'2022 [[Paper](https://arxiv.org/abs/2103.15375)]
[[Code](https://github.com/shashankvkt/AlignMixup_CVPR22)]
   <details close>
   <summary>AlignMix Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204574666-fcb694b0-c6bf-438a-bbd0-020635fe4b51.png" /></p>
   </details>

* **AutoMix: Unveiling the Power of Mixup for Stronger Classifiers**<br>
*Zicheng Liu, Siyuan Li, Di Wu, Zihan Liu, Zhiyuan Chen, Lirong Wu, Stan Z. Li*<br>
ECCV'2022 [[Paper](https://arxiv.org/abs/2103.13027)]
[[Code](https://github.com/Westlake-AI/openmixup)]
   <details close>
   <summary>AutoMix Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/174272662-19ce57ad-7b08-4e73-81b1-3bb81fee2fe5.png" /></p>
   </details>

* **Boosting Discriminative Visual Representation Learning with Scenario-Agnostic Mixup**<br>
*Siyuan Li, Zicheng Liu, Di Wu, Zihan Liu, Stan Z. Li*<br>
Arxiv'2021 [[Paper](https://arxiv.org/abs/2111.15454)]
[[Code](https://github.com/Westlake-AI/openmixup)]
   <details close>
   <summary>SAMix Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/174272657-fb662377-b7c3-4faa-8d9b-ea6f1e08549e.png" /></p>
   </details>

* **ScoreNet: Learning Non-Uniform Attention and Augmentation for Transformer-Based Histopathological Image Classification**<br>
*Thomas Stegmüller, Behzad Bozorgtabar, Antoine Spahr, Jean-Philippe Thiran*<br>
Arxiv'2022 [[Paper](https://arxiv.org/abs/2202.07570)]
   <details close>
   <summary>ScoreMix Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204576297-e97ea9c4-ee17-4ec3-a672-3f088ededb72.png" /></p>
   </details>

* **RecursiveMix: Mixed Learning with History**<br>
*Lingfeng Yang, Xiang Li, Borui Zhao, Renjie Song, Jian Yang*<br>
NIPS'2022 [[Paper](https://arxiv.org/abs/2203.06844)]
[[Code](https://github.com/implus/RecursiveMix-pytorch)]
   <details close>
   <summary>RecursiveMix Framework</summary>
   <p align="center"><img width="95%" src="https://user-images.githubusercontent.com/44519745/204576092-5fd92410-c12a-4691-8f7b-01901445f2a4.png" /></p>
   </details>

* **Expeditious Saliency-guided Mix-up through Random Gradient Thresholding**<br>
*Remy Sun, Clement Masson, Gilles Henaff, Nicolas Thome, Matthieu Cord.*<br>
ICPR'2022 [[Paper](https://arxiv.org/abs/2205.10158)]
   <details close>
   <summary>SciMix Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204575557-ddc6644e-a5e4-49ae-a95d-59856cc99a25.png" /></p>
   </details>

* **TransformMix: Learning Transformation and Mixing Strategies for Sample-mixing Data Augmentation**<br>
*Tsz-Him Cheung, Dit-Yan Yeung.*<\br> 
OpenReview'2023 [[Paper](https://openreview.net/forum?id=-1vpxBUtP0B)]
   <details close>
   <summary>TransformMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204575385-0b2d7470-0ffd-4b6b-977b-ef28b1617954.png" /></p>
   </details>

* **GuidedMixup: An Efficient Mixup Strategy Guided by Saliency Maps**<br>
*Minsoo Kang, Suhyun Kim*<br>
AAAI'2023 [[Paper](https://arxiv.org/abs/2306.16612)]
   <details close>
   <summary>GuidedMixup Framework</summary>
   <p align="center"><img width="95%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/250170540-97434afd-790c-4148-81dc-ff7129ca3f7c.png" /></p>
   </details>

* **MixPro: Data Augmentation with MaskMix and Progressive Attention Labeling for Vision Transformer**<br>
*Qihao Zhao, Yangyu Huang, Wei Hu, Fan Zhang, Jun Liu*<br>
ICLR'2023 [[Paper](https://openreview.net/forum?id=dRjWsd3gwsm)]
[[Code](https://github.com/fistyee/MixPro)]
   <details close>
   <summary>MixPro Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/224795935-afb936b2-fc77-4018-a681-72887f96fa59.png" /></p>
   </details>

* **Expeditious Saliency-guided Mix-up through Random Gradient Thresholding**<br>
*Minh-Long Luu, Zeyi Huang, Eric P.Xing, Yong Jae Lee, Haohan Wang*<br>
2nd Practical-DL Workshop @ AAAI'23 [[Paper](https://arxiv.org/abs/2212.04875)]
[[Code](https://github.com/minhlong94/Random-Mixup)]
   <details close>
   <summary>R-Mix and R-LMix Framework</summary>
   <p align="center"><img width="85%" src="https://raw.githubusercontent.com/minhlong94/Random-Mixup/main/assets/Mixups.png" /></p>
   </details>

* **SMMix: Self-Motivated Image Mixing for Vision Transformers**<br>
*Mengzhao Chen, Mingbao Lin, ZhiHang Lin, Yuxin Zhang, Fei Chao, Rongrong Ji*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2212.12977)]
[[Code](https://github.com/chenmnz/smmix)]
   <details close>
   <summary>SMMix Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/213537624-7359689e-b5af-4db1-a4ad-07876dd44089.png" /></p>
   </details>

<!-- * **Teach me how to Interpolate a Myriad of Embeddings**<br> -->
* **Embedding Space Interpolation Beyond Mini-Batch, Beyond Pairs and Beyond Examples**<br>
*Shashanka Venkataramanan, Ewa Kijak, Laurent Amsaleg, Yannis Avrithis*<br>
NeurIPS'2023 [[Paper](https://arxiv.org/abs/2206.14868)]
   <details close>
   <summary>MultiMix Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/224786198-a85b76b9-0f8b-434d-be59-bbc41438aac9.png" /></p>
   </details>

* **GradSalMix: Gradient Saliency-Based Mix for Image Data Augmentation**<br>
*Tao Hong, Ya Wang, Xingwu Sun, Fengzong Lian, Zhanhui Kang, Jinwen Ma*<br>
ICME'2023 [[Paper](https://ieeexplore.ieee.org/abstract/document/10219625)]
   <details close>
   <summary>GradSalMix Framework</summary>
   <p align="center"><img width="75%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/264142659-b8e2eef2-f6ea-4b03-bc03-0d2c08212f3a.png" /></p>
   </details>

* **LGCOAMix: Local and Global Context-and-Object-Part-Aware Superpixel-Based Data Augmentation for Deep Visual Recognition**<br>
*Fadi Dornaika, Danyang Sun*<br>
TIP'2023 [[Paper](https://ieeexplore.ieee.org/document/10348509)]
[[Code](https://github.com/DanielaPlusPlus/LGCOAMix)]
   <details close>
   <summary>LGCOAMix Framework</summary>
   <p align="center"><img width="95%" src="https://github.com/Westlake-AI/Awesome-Mixup/assets/44519745/802bdd05-dd5d-4d84-a66a-18d0fe9f2fa9" /></p>
   </details>

* **Catch-Up Mix: Catch-Up Class for Struggling Filters in CNN**<br>
*Minsoo Kang, Minkoo Kang, Suhyun Kim*<br>
AAAI'2024 [[Paper](https://arxiv.org/abs/2401.13193)]
   <details close>
   <summary>Catch-Up-Mix Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/Westlake-AI/Awesome-Mixup/assets/44519745/459a5d2b-cc74-4bd0-8cd2-46313652a9b8" /></p>
   </details>

* **Adversarial AutoMixup**<br>
*Huafeng Qin, Xin Jin, Yun Jiang, Mounim A. El-Yacoubi, Xinbo Gao*<br>
ICLR'2024 [[Paper](https://arxiv.org/abs/2312.11954)]
[[Code](https://github.com/jinxins/adversarial-automixup)]
   <details close>
   <summary>AdAutoMix Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/Westlake-AI/openmixup/assets/44519745/cdf24e60-e262-4d69-88a6-d1544e625679" /></p>
   </details>

<p align="right">(<a href="#top">back to top</a>)</p>

### Label Mixup Methods

* **mixup: Beyond Empirical Risk Minimization**<br>
*Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz*<br>
ICLR'2018 [[Paper](https://arxiv.org/abs/1710.09412)]
[[Code](https://github.com/facebookresearch/mixup-cifar10)]

* **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features**<br>
*Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo*<br>
ICCV'2019 [[Paper](https://arxiv.org/abs/1905.04899)]
[[Code](https://github.com/clovaai/CutMix-PyTorch)]

* **Metamixup: Learning adaptive interpolation policy of mixup with metalearning**<br>
*Zhijun Mai, Guosheng Hu, Dexiong Chen, Fumin Shen, Heng Tao Shen*<br>
TNNLS'2021 [[Paper](https://arxiv.org/abs/1908.10059)]
   <details close>
   <summary>MetaMixup Framework</summary>
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/204576802-4aa83a66-61ac-40fd-8904-3b4f9eda62ef.png" /></p>
   </details>

* **Mixup Without Hesitation**<br>
*Hao Yu, Huanyu Wang, Jianxin Wu*<br>
ICIG'2022 [[Paper](https://arxiv.org/abs/2101.04342)]
[[Code](https://github.com/yuhao318/mwh)]

* **Combining Ensembles and Data Augmentation can Harm your Calibration**<br>
*Yeming Wen, Ghassen Jerfel, Rafael Muller, Michael W. Dusenberry, Jasper Snoek, Balaji Lakshminarayanan, Dustin Tran*<br>
ICLR'2021 [[Paper](https://arxiv.org/abs/2010.09875)]
[[Code](https://github.com/google/edward2/tree/main/experimental/marginalization_mixup)]
   <details close>
   <summary>CAMixup Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204577092-06b2c74a-47cc-44f5-8423-9f37b1d0cbdc.png" /></p>
   </details>

* **Combining Ensembles and Data Augmentation can Harm your Calibration**<br>
*Zihang Jiang, Qibin Hou, Li Yuan, Daquan Zhou, Yujun Shi, Xiaojie Jin, Anran Wang, Jiashi Feng*<br>
NIPS'2021 [[Paper](https://arxiv.org/abs/2104.10858)]
[[Code](https://github.com/zihangJiang/TokenLabeling)]
   <details close>
   <summary>TokenLabeling Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204577372-f679ab10-a65f-4319-9a40-8393c20ad0fa.png" /></p>
   </details>

* **Saliency Grafting: Innocuous Attribution-Guided Mixup with Calibrated Label Mixing**<br>
*Joonhyung Park, June Yong Yang, Jinwoo Shin, Sung Ju Hwang, Eunho Yang*<br>
AAAI'2022 [[Paper](https://arxiv.org/abs/2112.08796)]
   <details close>
   <summary>Saliency Grafting Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204577555-7ffe34ed-74c1-4dff-95e8-f5a9e385f50c.png" /></p>
   </details>

* **TransMix: Attend to Mix for Vision Transformers**<br>
*Jie-Neng Chen, Shuyang Sun, Ju He, Philip Torr, Alan Yuille, Song Bai*<br>
CVPR'2022 [[Paper](https://arxiv.org/abs/2111.09833)]
[[Code](https://github.com/Beckschen/TransMix)]
   <details close>
   <summary>TransMix Framework</summary>
   <p align="center"><img width="60%" src="https://user-images.githubusercontent.com/44519745/204577728-8d59ad5f-0204-4943-aae7-dca6c48022ce.png" /></p>
   </details>

* **GenLabel: Mixup Relabeling using Generative Models**<br>
*Jy-yong Sohn, Liang Shang, Hongxu Chen, Jaekyun Moon, Dimitris Papailiopoulos, Kangwook Lee*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2201.02354)]
   <details close>
   <summary>GenLabel Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204578083-3212ed98-6f1b-422b-8764-0276a65bce8e.png" /></p>
   </details>

* **Harnessing Hard Mixed Samples with Decoupled Regularizer**<br>
*Zicheng Liu, Siyuan Li, Ge Wang, Cheng Tan, Lirong Wu, Stan Z. Li*<br>
NIPS'2023 [[Paper](https://arxiv.org/abs/2203.10761)]
[[Code](https://github.com/Westlake-AI/openmixup)]
   <details close>
   <summary>DecoupleMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204578387-4be9567c-963a-4d2d-8c1f-c7c5ade527b8.png" /></p>
   </details>

* **TokenMix: Rethinking Image Mixing for Data Augmentation in Vision Transformers**<br>
*Jihao Liu, Boxiao Liu, Hang Zhou, Hongsheng Li, Yu Liu*<br>
ECCV'2022 [[Paper](https://arxiv.org/abs/2207.08409)]
[[Code](https://github.com/Sense-X/TokenMix)]
   <details close>
   <summary>TokenMix Framework</summary>
   <p align="center"><img width="70%" src="https://user-images.githubusercontent.com/44519745/204578736-7b2dd349-7214-4d49-ade8-30b1caa2f1ea.png" /></p>
   </details>

* **Optimizing Random Mixup with Gaussian Differential Privacy**<br>
*Donghao Li, Yang Cao, Yuan Yao*<br>
arXiv'2022 [[Paper](https://arxiv.org/abs/2202.06467)]

* **TokenMixup: Efficient Attention-guided Token-level Data Augmentation for Transformers**<br>
*Hyeong Kyu Choi, Joonmyung Choi, Hyunwoo J. Kim*<br>
NIPS'2022 [[Paper](https://arxiv.org/abs/2210.07562)]
[[Code](https://github.com/mlvlab/TokenMixup)]
   <details close>
   <summary>TokenMixup Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/204578884-b9d7d466-b26b-4e4b-8a23-22199a6dca26.png" /></p>
   </details>

* **Token-Label Alignment for Vision Transformers**<br>
*Han Xiao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, Jiwen Lu*<br>
arXiv'2022 [[Paper](https://arxiv.org/abs/2210.06455)]
[[Code](https://github.com/Euphoria16/TL-Align)]
   <details close>
   <summary>TL-Align Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204579080-3b7c9352-8fb3-49bd-99f5-ce4f72d722d8.png" /></p>
   </details>

* **LUMix: Improving Mixup by Better Modelling Label Uncertainty**<br>
*Shuyang Sun, Jie-Neng Chen, Ruifei He, Alan Yuille, Philip Torr, Song Bai*<br>
arXiv'2022 [[Paper](https://arxiv.org/abs/2211.15846)]
[[Code](https://github.com/kevin-ssy/LUMix)]
   <details close>
   <summary>LUMix Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/205531445-dc4b7790-e0b7-4c41-b9d2-708efa5e4198.png" /></p>
   </details>

* **MixupE: Understanding and Improving Mixup from Directional Derivative Perspective**<br>
*Vikas Verma, Sarthak Mittal, Wai Hoh Tang, Hieu Pham, Juho Kannala, Yoshua Bengio, Arno Solin, Kenji Kawaguchi*<br>
UAI'2023 [[Paper](https://arxiv.org/abs/2212.13381)]
[[Code](https://github.com/onehuster/mixupe)]
   <details close>
   <summary>MixupE Framework</summary>
   <p align="center"><img width="85%" src="https://user-images.githubusercontent.com/44519745/209991074-3dd41cdf-4e64-42e2-8bf4-ebc60e8212d0.png" /></p>
   </details>

* **Infinite Class Mixup**<br>
*Thomas Mensink, Pascal Mettes*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2305.10293)]
   <details close>
   <summary>IC-Mixup Framework</summary>
   <p align="center"><img width="60%" src="https://github.com/Westlake-AI/openmixup/assets/44519745/5082d12b-8cb0-4a3e-8258-c0dbf78e3159" /></p>
   </details>

* **Semantic Equivariant Mixup**<br>
*Zongbo Han, Tianchi Xie, Bingzhe Wu, Qinghua Hu, Changqing Zhang*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2308.06451)]
   <details close>
   <summary>SEM Framework</summary>
   <p align="center"><img width="75%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/264147910-bbe70d79-b521-4de2-8c5e-251301cfc6ad.png" /></p>
   </details>

* **RankMixup: Ranking-Based Mixup Training for Network Calibration**<br>
*Jongyoun Noh, Hyekang Park, Junghyup Lee, Bumsub Ham*<br>
ICCV'2023 [[Paper](https://arxiv.org/abs/2308.11990)]
[[Code](https://cvlab.yonsei.ac.kr/projects/RankMixup)]
   <details close>
   <summary>RankMixup Framework</summary>
   <p align="center"><img width="60%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/264144742-051304d2-4f64-4bd7-9e70-12074c2215e4.png" /></p>
   </details>

* **G-Mix: A Generalized Mixup Learning Framework Towards Flat Minima**<br>
*Xingyu Li, Bo Tang*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2308.03236)]
   <details close>
   <summary>G-Mix Framework</summary>
   <p align="center"><img width="70%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/264155347-1a94f3d1-a9d8-46e1-bb32-bed0bfb449ca.png" /></p>
   </details>


<p align="right">(<a href="#top">back to top</a>)</p>

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

* **i-MAE: Are Latent Representations in Masked Autoencoders Linearly Separable**<br>
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

* **Guarding Barlow Twins Against Overfitting with Mixed Samples**<br>
*Wele Gedara Chaminda Bandara, Celso M. De Melo, Vishal M. Patel*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2312.02151)]
[[Code](https://github.com/wgcban/mix-bt)]
   <details close>
   <summary>PatchMix Framework</summary>
   <p align="center"><img width="80%" src="https://github.com/Westlake-AI/Awesome-Mixup/assets/44519745/342f7dba-ccff-4b22-99e7-3d1d965ea1ee" /></p>
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

* **Epsilon Consistent Mixup: Structural Regularization with an Adaptive Consistency-Interpolation Tradeoff**<br>
*Vincent Pisztora, Yanglan Ou, Xiaolei Huang, Francesca Chiaromonte, Jia Li*<br>
ArXiv'2021 [[Paper](https://arxiv.org/abs/2104.09452)]
   <details close>
   <summary>Epsilon Consistent Mixup (ϵmu) Framework</summary>
   <p align="center"><img width="50%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/283528510-1f3b643c-0edd-416e-9979-110f3d2be6b6.png" /></p>
   </details>

* **Unleashing the Power of Contrastive Self-Supervised Visual Models via Contrast-Regularized Fine-Tuning**<br>
*Yifan Zhang, Bryan Hooi, Dapeng Hu, Jian Liang, Jiashi Feng*<br>
NIPS'2021 [[Paper](https://arxiv.org/abs/2102.06605)]
[[Code](https://github.com/vanint/core-tuning)]
   <details close>
   <summary>Core-Tuning Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204580135-ed6ba8b7-b69c-4683-90f0-9aa9cdd530bc.png" /></p>
   </details>

* **MUM: Mix Image Tiles and UnMix Feature Tiles for Semi-Supervised Object Detection**<br>
*JongMok Kim, Jooyoung Jang, Seunghyeon Seo, Jisoo Jeong, Jongkeun Na, Nojun Kwak*<br>
CVPR'2022 [[Paper](https://user-images.githubusercontent.com/44519745/225082975-4143e7f5-8873-433c-ab6f-6caa615f7120.png)]
[[Code](https://github.com/jongmokkim/mix-unmix)]
   <details close>
   <summary>MUM Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/225082975-4143e7f5-8873-433c-ab6f-6caa615f7120.png" /></p>
   </details>

* **Harnessing Hard Mixed Samples with Decoupled Regularizer**<br>
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

* **Mixed Pseudo Labels for Semi-Supervised Object Detection**<br>
*Zeming Chen, Wenwei Zhang, Xinjiang Wang, Kai Chen, Zhi Wang*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2312.07006)]
[[Code](https://github.com/czm369/mixpl)]
   <details close>
   <summary>MixPL Framework</summary>
   <p align="center"><img width="90%" src="https://github.com/Westlake-AI/Awesome-Mixup/assets/44519745/c1ac594f-18bb-465b-b0e7-e96249231e2c" /></p>
   </details>

* **PCLMix: Weakly Supervised Medical Image Segmentation via Pixel-Level Contrastive Learning and Dynamic Mix Augmentation**<br>
*Yu Lei, Haolun Luo, Lituan Wang, Zhenwei Zhang, Lei Zhang*<br>
ArXiv'2024 [[Paper](https://arxiv.org/abs/2405.06288)]
[[Code](https://github.com/Torpedo2648/PCLMix)]
   <details close>
   <summary>PCLMix Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/Lupin1998/Awesome-MIM/assets/44519745/e41b7e65-962b-49fa-b1be-6e789e8b244e" /></p>
   </details>

## Mixup for Regression

* **RegMix: Data Mixing Augmentation for Regression**<br>
*Seong-Hyeon Hwang, Steven Euijong Whang*<br>
ArXiv'2021 [[Paper](https://arxiv.org/abs/2106.03374)]

* **C-Mixup: Improving Generalization in Regression**<br>
*Huaxiu Yao, Yiping Wang, Linjun Zhang, James Zou, Chelsea Finn*<br>
NeurIPS'2022 [[Paper](https://arxiv.org/abs/2210.05775)]
[[Code](https://github.com/huaxiuyao/C-Mixup)]

* **ExtraMix: Extrapolatable Data Augmentation for Regression using Generative Models**<br>
*Kisoo Kwon, Kuhwan Jeong, Sanghyun Park, Sangha Park, Hoshik Lee, Seung-Yeon Kwak, Sungmin Kim, Kyunghyun Cho*<br>
OpenReview'2022 [[Paper](https://openreview.net/forum?id=NgEuFT-SIgI)]

* **Anchor Data Augmentation**<br>
*Nora Schneider, Shirin Goshtasbpour, Fernando Perez-Cruz*<br>
NeurIPS'2023 [[Paper](https://arxiv.org/abs/2311.06965)]

* **Rank-N-Contrast: Learning Continuous Representations for Regression**<br>
*Kaiwen Zha, Peng Cao, Jeany Son, Yuzhe Yang, Dina Katabi*<br>
NeurIPS'2023 [[Paper](https://arxiv.org/abs/2210.01189)]
[[Code](https://github.com/kaiwenzha/Rank-N-Contrast)]

* **Mixup Your Own Pairs**<br>
*Yilei Wu, Zijian Dong, Chongyao Chen, Wangchunshu Zhou, Juan Helen Zhou*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2309.16633)]
[[Code](https://github.com/yilei-wu/supremix)]
   <details close>
   <summary>SupReMix Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/283530899-d69f9dca-04ee-487f-bef0-4d16f63d38f8.png" /></p>
   </details>

* **Tailoring Mixup to Data using Kernel Warping functions**<br>
*Quentin Bouniot, Pavlo Mozharovskyi, Florence d'Alché-Buc*<br>
ArXiv'2023 [[Paper](https://arxiv.org/abs/2311.01434)]
[[Code](https://github.com/ENSTA-U2IS/torch-uncertainty)]
   <details close>
   <summary>SupReMix Framework</summary>
   <p align="center"><img width="90%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/285024114-9e741976-bd59-46dc-bd57-dc498079a6bb.png" /></p>
   </details>

* **OmniMixup: Generalize Mixup with Mixing-Pair Sampling Distribution**<br>
*Anonymous*<br>
Openreview'2023 [[Paper](https://openreview.net/forum?id=6Uc7Fgwrsm)]

* **Augment on Manifold: Mixup Regularization with UMAP**<br>
*Yousef El-Laham, Elizabeth Fons, Dillon Daudert, Svitlana Vyetrenko*<br>
ICASSP'2024 [[Paper](https://arxiv.org/abs/2210.01189)]

## Mixup for Robustness

* **Mixup as directional adversarial training**<br>
*Guillaume P. Archambault, Yongyi Mao, Hongyu Guo, Richong Zhang*<br>
NeurIPS'2019 [[Paper](https://arxiv.org/abs/1906.06875)]
[[Code](https://github.com/mixupAsDirectionalAdversarial/mixup_as_dat)]

* **Mixup Inference: Better Exploiting Mixup to Defend Adversarial Attacks**<br>
*Tianyu Pang, Kun Xu, Jun Zhu*<br>
ICLR'2020 [[Paper](https://arxiv.org/abs/1909.11515)]
[[Code](https://github.com/P2333/Mixup-Inference)]

* **Addressing Neural Network Robustness with Mixup and Targeted Labeling Adversarial Training**<br>
*Alfred Laugros, Alice Caplier, Matthieu Ospici*<br>
ECCV'2020 [[Paper](https://arxiv.org/abs/2008.08384)]

* **Mixup Training as the Complexity Reduction**<br>
*Masanari Kimura*<br>
OpenReview'2021 [[Paper](https://openreview.net/forum?id=xvWZQtxI7qq)]

* **Adversarial Vertex Mixup: Toward Better Adversarially Robust Generalization**<br>
*Saehyung Lee, Hyungyu Lee, Sungroh Yoon*<br>
CVPR'2020 [[Paper](https://arxiv.org/abs/2003.02484)]
[[Code](https://github.com/Saehyung-Lee/cifar10_challenge)]

* **MixACM: Mixup-Based Robustness Transfer via Distillation of Activated Channel Maps**<br>
*Muhammad Awais, Fengwei Zhou, Chuanlong Xie, Jiawei Li, Sung-Ho Bae, Zhenguo Li*<br>
NeurIPS'2021 [[Paper](https://arxiv.org/abs/2111.05073)]

* **On the benefits of defining vicinal distributions in latent space**<br>
*Puneet Mangla, Vedant Singh, Shreyas Jayant Havaldar, Vineeth N Balasubramanian*<br>
CVPRW'2021 [[Paper](https://arxiv.org/abs/2003.06566)]

## Low-level Vision

* **Robust Image Denoising through Adversarial Frequency Mixup**<br>
*Donghun Ryou, Inju Ha, Hyewon Yoo, Dongwan Kim, Bohyung Han*<br>
CVPR'2024 [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Ryou_Robust_Image_Denoising_through_Adversarial_Frequency_Mixup_CVPR_2024_paper.pdf)]
[[Code](https://github.com/dhryougit/AFM)]

<p align="right">(<a href="#top">back to top</a>)</p>

## Mixup for Multi-modality

* **MixGen: A New Multi-Modal Data Augmentation**<br>
*Xiaoshuai Hao, Yi Zhu, Srikar Appalaraju, Aston Zhang, Wanqian Zhang, Bo Li, Mu Li*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2206.08358)]
[[Code](https://github.com/amazon-research/mix-generation)]

* **VLMixer: Unpaired Vision-Language Pre-training via Cross-Modal CutMix**<br>
*Teng Wang, Wenhao Jiang, Zhichao Lu, Feng Zheng, Ran Cheng, Chengguo Yin, Ping Luo*<br>
arXiv'2022 [[Paper](https://arxiv.org/abs/2206.08919)]

* **Geodesic Multi-Modal Mixup for Robust Fine-Tuning**<br>
*Changdae Oh, Junhyuk So, Hoyoon Byun, YongTaek Lim, Minchul Shin, Jong-June Jeon, Kyungwoo Song*<br>
NeurIPS'2023 [[Paper](https://arxiv.org/abs/2203.03897)]
[[Code](https://github.com/changdaeoh/multimodal-mixup)]

* **PowMix: A Versatile Regularizer for Multimodal Sentiment Analysis**<br>
*Efthymios Georgiou, Yannis Avrithis, Alexandros Potamianos*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2312.12334)]
   <details close>
   <summary>PowMix Framework</summary>
   <p align="center"><img width="85%" src="https://github.com/Westlake-AI/openmixup/assets/44519745/272e44b8-1872-4cdf-9a69-71e48cc428cd" /></p>
   </details>

## Analysis of Mixup

* **On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks**<br>
*Sunil Thulasidasan, Gopinath Chennupati, Jeff Bilmes, Tanmoy Bhattacharya, Sarah Michalak*<br>
NeurIPS'2019 [[Paper](https://arxiv.org/abs/1905.11001)]
[[Code](https://github.com/paganpasta/onmixup)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204825679-ee39834f-7346-4465-b58e-c4909dec767f.png" /></p>
   </details>

* **On Mixup Regularization**<br>
*Luigi Carratino, Moustapha Cissé, Rodolphe Jenatton, Jean-Philippe Vert*<br>
ArXiv'2020 [[Paper](https://arxiv.org/abs/2006.06049)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204826066-62a66221-b023-462d-a9b7-89bbd7614425.png" /></p>
   </details>

* **How Does Mixup Help With Robustness and Generalization?**<br>
*Linjun Zhang, Zhun Deng, Kenji Kawaguchi, Amirata Ghorbani, James Zou*<br>
ICLR'2021 [[Paper](https://arxiv.org/abs/2010.04819)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204826319-9ae701af-ad99-4780-85a6-492839c5bcbf.png" /></p>
   </details>

* **Towards Understanding the Data Dependency of Mixup-style Training**<br>
*Muthu Chidambaram, Xiang Wang, Yuzheng Hu, Chenwei Wu, Rong Ge*<br>
ICLR'2022 [[Paper](https://openreview.net/pdf?id=ieNJYujcGDO)]
[[Code](https://github.com/2014mchidamb/Mixup-Data-Dependency)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="65%" src="https://user-images.githubusercontent.com/44519745/204826671-e012b918-5a14-4b5f-929b-b2f079b79d3f.png" /></p>
   </details>

* **When and How Mixup Improves Calibration**<br>
*Linjun Zhang, Zhun Deng, Kenji Kawaguchi, James Zou*<br>
ICML'2022 [[Paper](https://arxiv.org/abs/2102.06289)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="50%" src="https://user-images.githubusercontent.com/44519745/204827323-b854f0a0-803f-46f8-9b74-44970998b311.png" /></p>
   </details>

* **Over-Training with Mixup May Hurt Generalization**<br>
*Zixuan Liu, Ziqiao Wang, Hongyu Guo, Yongyi Mao*<br>
ICLR'2023 [[Paper](https://openreview.net/forum?id=JmkjrlVE-DG)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/216831436-704b0427-61a0-42ff-b4e6-ab361be8b634.png" /></p>
   </details>

* **Provable Benefit of Mixup for Finding Optimal Decision Boundaries**<br>
*Junsoo Oh, Chulhee Yun*<br>
ICML'2023 [[Paper](https://chulheeyun.github.io/publication/oh2023provable/)]

* **On the Pitfall of Mixup for Uncertainty Calibration**<br>
*Deng-Bao Wang, Lanqing Li, Peilin Zhao, Pheng-Ann Heng, Min-Ling Zhang*<br>
CVPR'2023 [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_On_the_Pitfall_of_Mixup_for_Uncertainty_Calibration_CVPR_2023_paper.pdf)]

* **Understanding the Role of Mixup in Knowledge Distillation: An Empirical Study**<br>
*Hongjun Choi, Eun Som Jeon, Ankita Shukla, Pavan Turaga*<br>
WACV'2023 [[Paper](https://arxiv.org/abs/2211.03946)]
[[Code](https://github.com/hchoi71/mix-kd)]

* **Analyzing Effects of Mixed Sample Data Augmentation on Model Interpretability**<br>
*Soyoun Won, Sung-Ho Bae, Seong Tae Kim*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2303.14608)]

* **Selective Mixup Helps with Distribution Shifts, But Not (Only) because of Mixup**<br>
*Damien Teney, Jindong Wang, Ehsan Abbasnejad*<br>
arXiv'2023 [[Paper](https://arxiv.org/abs/2305.16817)]

<p align="right">(<a href="#top">back to top</a>)</p>

## Natural Language Processing

* **Augmenting Data with Mixup for Sentence Classification: An Empirical Study**<br>
*Hongyu Guo, Yongyi Mao, Richong Zhang*<br>
arXiv'2019 [[Paper](https://arxiv.org/abs/1905.08941)]
[[Code](https://github.com/dsfsi/textaugment)]

* **Mixup-Transformer: Dynamic Data Augmentation for NLP Tasks**<br>
*Lichao Sun, Congying Xia, Wenpeng Yin, Tingting Liang, Philip S. Yu, Lifang He*<br>
COLING'2020 [[Paper](https://arxiv.org/abs/2010.02394)]

* **Calibrated Language Model Fine-Tuning for In- and Out-of-Distribution Data**<br>
*Lingkai Kong, Haoming Jiang, Yuchen Zhuang, Jie Lyu, Tuo Zhao, Chao Zhang*<br>
EMNLP'2020 [[Paper](https://arxiv.org/abs/2010.11506)]
[[Code](https://github.com/Lingkai-Kong/Calibrated-BERT-Fine-Tuning)]

* **Augmenting NLP Models using Latent Feature Interpolations**<br>
*Amit Jindal, Arijit Ghosh Chowdhury, Aniket Didolkar, Di Jin, Ramit Sawhney, Rajiv Ratn Shah*<br>
COLING'2020 [[Paper](https://aclanthology.org/2020.coling-main.611/)]

* **MixText: Linguistically-informed Interpolation of Hidden Space for Semi-Supervised Text Classification**<br>
*Jiaao Chen, Zichao Yang, Diyi Yang*<br>
ACL'2020 [[Paper](https://arxiv.org/abs/2004.12239)]
[[Code](https://github.com/GT-SALT/MixText)]

* **TreeMix: Compositional Constituency-based Data Augmentation for Natural Language Understanding**<br>
*Le Zhang, Zichao Yang, Diyi Yang*<br>
NAALC'2022 [[Paper](https://arxiv.org/abs/2205.06153)]
[[Code](https://github.com/magiccircuit/treemix)]

* **STEMM: Self-learning with Speech-text Manifold Mixup for Speech Translation**<br>
*Qingkai Fang, Rong Ye, Lei Li, Yang Feng, Mingxuan Wang*<br>
ACL'2022 [[Paper](https://arxiv.org/abs/2010.02394)]
[[Code](https://github.com/ictnlp/STEMM)]

* **Enhancing Cross-lingual Transfer by Manifold Mixup**<br>
*Huiyun Yang, Huadong Chen, Hao Zhou, Lei Li*<br>
ICLR'2022 [[Paper](https://arxiv.org/abs/2205.04182)]
[[Code](https://github.com/yhy1117/x-mixup)]

## Graph Representation Learning

* **Fused Gromov-Wasserstein Graph Mixup for Graph-level Classifications**<br>
*Xinyu Ma, Xu Chu, Yasha Wang, Yang Lin, Junfeng Zhao, Liantao Ma, Wenwu Zhu*<br>
NeurIPS'2023 [[Paper](https://arxiv.org/abs/2306.15963)]
[[code](https://github.com/ArthurLeoM/FGWMixup)]

* **G-Mixup: Graph Data Augmentation for Graph Classification**<br>
*Xiaotian Han, Zhimeng Jiang, Ninghao Liu, Xia Hu*<br>
NeurIPS'2023 [[Paper](https://arxiv.org/abs/2202.07179)]

<p align="right">(<a href="#top">back to top</a>)</p>

## Survey

* **A survey on Image Data Augmentation for Deep Learning**<br>
*Connor Shorten and Taghi Khoshgoftaar*<br>
Journal of Big Data'2019 [[Paper](https://www.researchgate.net/publication/334279066_A_survey_on_Image_Data_Augmentation_for_Deep_Learning)]

* **An overview of mixing augmentation methods and augmentation strategies**<br>
*Dominik Lewy and Jacek Ma ́ndziuk*<br>
Artificial Intelligence Review'2022 [[Paper](https://link.springer.com/article/10.1007/s10462-022-10227-z)]

* **Image Data Augmentation for Deep Learning: A Survey**<br>
*Suorong Yang, Weikang Xiao, Mengcheng Zhang, Suhan Guo, Jian Zhao, Furao Shen*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2204.08610)]

* **A Survey of Mix-based Data Augmentation: Taxonomy, Methods, Applications, and Explainability**<br>
*Chengtai Cao, Fan Zhou, Yurou Dai, Jianping Wang*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2212.10888)]
[[Code](https://github.com/ChengtaiCao/Awesome-Mix)]

* **A Survey of Automated Data Augmentation for Image Classification: Learning to Compose, Mix, and Generate**<br>
*Tsz-Him Cheung, Dit-Yan Yeung*<br>
TNNLS'2023 [[Paper](https://ieeexplore.ieee.org/abstract/document/10158722)]

* **Survey: Image Mixing and Deleting for Data Augmentation**<br>
*Humza Naveed, Saeed Anwar, Munawar Hayat, Kashif Javed, Ajmal Mian*<br>
Engineering Applications of Artificial Intelligence'2024 [[Paper](https://arxiv.org/abs/2106.07085)]

## Benchmark

* **OpenMixup: A Comprehensive Mixup Benchmark for Visual Classification**<br>
*Siyuan Li, Zedong Wang, Zicheng Liu, Di Wu, Cheng Tan, Weiyang Jin, Stan Z. Li*<br>
ArXiv'2022 [[Paper](https://arxiv.org/abs/2209.04851)]
[[Code](https://github.com/Westlake-AI/openmixup)]

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
