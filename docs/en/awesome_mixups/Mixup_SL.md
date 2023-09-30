# Awesome Mixup Methods for Supervised Learning

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![GitHub stars](https://img.shields.io/github/stars/Westlake-AI/openmixup?color=blue) ![GitHub forks](https://img.shields.io/github/forks/Westlake-AI/openmixup?color=yellow&label=Fork)

**We summarize fundamental mixup methods proposed for supervised visual representation learning from two aspects: *sample mixup policy* and *label mixup policy*. Then, we summarize mixup techniques used in downstream tasks.**
The list of awesome mixup methods is summarized in chronological order and is on updating. And we will add more papers according to [Awesome-Mix](https://github.com/ChengtaiCao/Awesome-Mix).

* To find related papers and their relationships, check out [Connected Papers](https://www.connectedpapers.com/), which visualizes the academic field in a graph representation.
* To export BibTeX citations of papers, check out [ArXiv](https://arxiv.org/) or [Semantic Scholar](https://www.semanticscholar.org/) of the paper for professional reference formats.

## Table of Contents

  - [Sample Mixup Methods](#sample-mixup-methods)
    + [Pre-defined Policies](#pre-defined-policies)
    + [Saliency-guided Policies](#saliency-guided-policies)
  - [Label Mixup Methods](#label-mixup-methods)
  - [Analysis of Mixup](#analysis-of-mixup)
  - [Survey](#survey)
  - [Contribution](#contribution)
  - [Related Project](#related-project)

## Sample Mixup Methods

### Pre-defined Policies

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

<p align="right">(<a href="#top">back to top</a>)</p>

### Saliency-guided Policies

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

* **Teach me how to Interpolate a Myriad of Embeddings**<br>
*Shashanka Venkataramanan, Ewa Kijak, Laurent Amsaleg, Yannis Avrithis*<br>
Arxiv'2022 [[Paper](https://arxiv.org/abs/2206.14868)]
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

<p align="right">(<a href="#top">back to top</a>)</p>

## Label Mixup Methods

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

* **Decoupled Mixup for Data-efficient Learning**<br>
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
arXiv'2022 [[Paper](https://arxiv.org/abs/2212.13381)]
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

## Analysis of Mixup

* Sunil Thulasidasan, Gopinath Chennupati, Jeff Bilmes, Tanmoy Bhattacharya, Sarah Michalak.
   - On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks. [[NIPS'2019](https://arxiv.org/abs/1905.11001)] [[code](https://github.com/paganpasta/onmixup)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/204825679-ee39834f-7346-4465-b58e-c4909dec767f.png" /></p>
   </details>
* Luigi Carratino, Moustapha Cissé, Rodolphe Jenatton, Jean-Philippe Vert.
   - On Mixup Regularization. [[ArXiv'2020](https://arxiv.org/abs/2006.06049)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="75%" src="https://user-images.githubusercontent.com/44519745/204826066-62a66221-b023-462d-a9b7-89bbd7614425.png" /></p>
   </details>
* Linjun Zhang, Zhun Deng, Kenji Kawaguchi, Amirata Ghorbani, James Zou.
   - How Does Mixup Help With Robustness and Generalization? [[ICLR'2021](https://arxiv.org/abs/2010.04819)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="80%" src="https://user-images.githubusercontent.com/44519745/204826319-9ae701af-ad99-4780-85a6-492839c5bcbf.png" /></p>
   </details>
* Muthu Chidambaram, Xiang Wang, Yuzheng Hu, Chenwei Wu, Rong Ge.
   - Towards Understanding the Data Dependency of Mixup-style Training. [[ICLR'2022](https://openreview.net/pdf?id=ieNJYujcGDO)] [[code](https://github.com/2014mchidamb/Mixup-Data-Dependency)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="65%" src="https://user-images.githubusercontent.com/44519745/204826671-e012b918-5a14-4b5f-929b-b2f079b79d3f.png" /></p>
   </details>
* Linjun Zhang, Zhun Deng, Kenji Kawaguchi, James Zou.
   - When and How Mixup Improves Calibration. [[ICML'2022](https://arxiv.org/abs/2102.06289)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="50%" src="https://user-images.githubusercontent.com/44519745/204827323-b854f0a0-803f-46f8-9b74-44970998b311.png" /></p>
   </details>
* Zixuan Liu, Ziqiao Wang, Hongyu Guo, Yongyi Mao.
   - Over-Training with Mixup May Hurt Generalization. [[ICLR'2023](https://openreview.net/forum?id=JmkjrlVE-DG)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="90%" src="https://user-images.githubusercontent.com/44519745/216831436-704b0427-61a0-42ff-b4e6-ab361be8b634.png" /></p>
   </details>
* Junsoo Oh, Chulhee Yun.
   - Provable Benefit of Mixup for Finding Optimal Decision Boundaries. [[ICML'2023](https://chulheeyun.github.io/publication/oh2023provable/)]
* Deng-Bao Wang, Lanqing Li, Peilin Zhao, Pheng-Ann Heng, Min-Ling Zhang.
   - On the Pitfall of Mixup for Uncertainty Calibration. [[CVPR'2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_On_the_Pitfall_of_Mixup_for_Uncertainty_Calibration_CVPR_2023_paper.pdf)]
* Hongjun Choi, Eun Som Jeon, Ankita Shukla, Pavan Turaga.
   - Understanding the Role of Mixup in Knowledge Distillation: An Empirical Study. [[WACV'2023](https://arxiv.org/abs/2211.03946)] [[code](https://github.com/hchoi71/mix-kd)]
* Soyoun Won, Sung-Ho Bae, Seong Tae Kim.
   - Analyzing Effects of Mixed Sample Data Augmentation on Model Interpretability. [[arXiv'2023](https://arxiv.org/abs/2303.14608)]

<p align="right">(<a href="#top">back to top</a>)</p>

## Survey

* **A survey on Image Data Augmentation for Deep Learning**<br>
*Connor Shorten and Taghi Khoshgoftaar*<br>
Journal of Big Data'2019 [[Paper](https://www.researchgate.net/publication/334279066_A_survey_on_Image_Data_Augmentation_for_Deep_Learning)]

* **Survey: Image Mixing and Deleting for Data Augmentation**<br>
*Humza Naveed, Saeed Anwar, Munawar Hayat, Kashif Javed, Ajmal Mian*<br>
ArXiv'2021 [[Paper](https://arxiv.org/abs/2106.07085)]
[[Code](https://github.com/humza909/survery-image-mixing-and-deleting-for-data-augmentation)]

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
