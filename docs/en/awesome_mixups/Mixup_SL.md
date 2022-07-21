# Awesome Mixup Methods for Supervised Learning

 [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![GitHub stars](https://img.shields.io/github/stars/Westlake-AI/openmixup?color=green) ![visitors](https://visitor-badge.glitch.me/badge?page_id=Westlake-AI/openmixup)

**We summarize mixup methods proposed for supervised visual representation learning from two aspects: *sample mixup policy* and *label mixup policy*.**
We are working on a survey of mixup methods. The list of awesome mixup methods is summarized in chronological order and is on updating.

## Sample Mixup Methods

### Pre-defined Policies

1. **MixUp**, [[ICLR'2018](https://arxiv.org/abs/1710.09412)] [[code](https://github.com/facebookresearch/mixup-cifar10)]
   mixup: Beyond Empirical Risk Minimization.
2. **AdaMixup**, [[AAAI'2019](https://arxiv.org/abs/1710.09412)]
   MixUp as Locally Linear Out-Of-Manifold Regularization.
3. **CutMix**, [[ICCV'2019](https://arxiv.org/abs/1905.04899)] [[code](https://github.com/clovaai/CutMix-PyTorch)]
   CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.
4. **ManifoldMix**, [[ICML'2019](https://arxiv.org/abs/1806.05236)] [[code](https://github.com/vikasverma1077/manifold_mixup)]
   Manifold Mixup: Better Representations by Interpolating Hidden States.
5. **FMix**, [[Arixv'2020](https://arxiv.org/abs/2002.12047)] [[code](https://github.com/ecs-vlc/FMix)]
   FMix: Enhancing Mixed Sample Data Augmentation.
6. **SmoothMix**, [[CVPRW'2020](https://arxiv.org/abs/2002.12047)] [[code](https://github.com/Westlake-AI/openmixup)]
   SmoothMix: a Simple Yet Effective Data Augmentation to Train Robust Classifiers.
7. **PatchUp**, [[Arxiv'2020](https://arxiv.org/abs/2006.07794)] [[code](https://github.com/chandar-lab/PatchUp)]
   PatchUp: A Regularization Technique for Convolutional Neural Networks.
8. **GridMixup**, [[Pattern Recognition'2021](https://www.sciencedirect.com/science/article/pii/S0031320320303976)] [[code](https://github.com/IlyaDobrynin/GridMixup)]
   GridMix: Strong regularization through local context mapping.
9. **SmoothMix**, [[CVPRW'2020](https://openaccess.thecvf.com/content_CVPRW_2020/html/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.html)]
   SmoothMix: A Simple Yet Effective Data Augmentation to Train Robust Classifiers.
10. **ResizeMix**, [[Arixv'2020](https://arxiv.org/abs/2012.11101)] [[code](https://github.com/Westlake-AI/openmixup)]
    ResizeMix: Mixing Data with Preserved Object Information and True Labels.
11. **FocusMix**, [[ICTC'2020](https://ieeexplore.ieee.org/abstract/document/9289404)]
    Where to Cut and Paste: Data Regularization with Selective Features.
12. **AugMix**, [[ICLR'2020](https://arxiv.org/abs/1912.02781)] [[code](https://github.com/google-research/augmix)]
    AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty.
13. **DJMix**, [[Arxiv'2021](https://openreview.net/pdf?id=0n3BaVlNsHI)]
    DJMix: Unsupervised Task-agnostic Augmentation for Improving Robustness.
14. **PixMix**, [[Arxiv'2021](https://arxiv.org/abs/2112.05135)] [[code](https://github.com/andyzoujm/pixmix)]
    PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures.
15. **StyleMix**, [[CVPR'2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_StyleMix_Separating_Content_and_Style_for_Enhanced_Data_Augmentation_CVPR_2021_paper.pdf)] [[code](https://github.com/alsdml/StyleMix)]
    StyleMix: Separating Content and Style for Enhanced Data Augmentation.
16. **MixStyle**, [[ICLR'2021](https://openreview.net/forum?id=6xHJ37MVxxp)] [[code](https://github.com/KaiyangZhou/mixstyle-release)]
    Domain Generalization with MixStyle.
17. **MoEx**, [[CVPR'2021](https://arxiv.org/abs/2002.11102)] [[code](https://github.com/Boyiliee/MoEx)]
    On Feature Normalization and Data Augmentation.
18. **LocalMix**, [[AISTATS'2021](https://arxiv.org/abs/2201.04368)]
    Preventing Manifold Intrusion with Locality: Local Mixup.

### Saliency-guided Policies

1. **SaliencyMix**, [[ICLR'2021](https://arxiv.org/abs/2006.01791)] [[code](https://github.com/SaliencyMix/SaliencyMix)]
   SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization.
2. **AttentiveMix**, [[ICASSP'2020](https://arxiv.org/abs/2003.13048)] [[code](https://github.com/xden2331/attentive_cutmix)]
   Attentive CutMix: An Enhanced Data Augmentation Approach for Deep Learning Based Image Classification.
3. **SnapMix**, [[AAAI'2021](https://arxiv.org/abs/2012.04846)] [[code](https://github.com/Shaoli-Huang/SnapMix)]
   SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data.
4. **AttributeMix**, [[Arxiv'2020](https://arxiv.org/abs/2004.02684)]
   Attribute Mix: Semantic Data Augmentation for Fine Grained Recognition.
5. **AutoMix**, [[ECCV'2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550630.pdf)]
   AutoMix: Mixup Networks for Sample Interpolation via Cooperative Barycenter Learning.
6. **PuzzleMix**, [[ICML'2020](https://arxiv.org/abs/2009.06962)] [[code](https://github.com/snu-mllab/PuzzleMix)]
   Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup.
7. **CoMixup**, [[ICLR'2021](https://openreview.net/forum?id=gvxJzw8kW4b)] [[code](https://github.com/snu-mllab/Co-Mixup)]
   Co-Mixup: Saliency Guided Joint Mixup with Supermodular Diversity.
8. **SuperMix**, [[CVPR'2021](https://arxiv.org/abs/2003.05034)] [[code](https://github.com/alldbi/SuperMix)]
   SuperMix: Supervising the Mixing Data Augmentation.
9. **PatchMix**, [[Arxiv'2021](https://arxiv.org/pdf/2106.09011.pdf)]
   Evolving Image Compositions for Feature Representation Learning.
10. **StackMix**, [[Arxiv'2021](https://arxiv.org/abs/2011.12618)]
   StackMix: A complementary Mix algorithm.
11. **AlignMix**, [[CVPR'2022](https://arxiv.org/abs/2103.15375)] [[code](https://github.com/shashankvkt/AlignMixup_CVPR22)]
   AlignMix: Improving representation by interpolating aligned features.
12. **AutoMix**, [[ECCV'2022](https://arxiv.org/abs/2103.13027)] [[code](https://github.com/Westlake-AI/openmixup)]
   AutoMix: Unveiling the Power of Mixup for Stronger Classifiers.
13. **SAMix**, [[Arxiv'2021](https://arxiv.org/abs/2111.15454)] [[code](https://github.com/Westlake-AI/openmixup)]
   Boosting Discriminative Visual Representation Learning with Scenario-Agnostic Mixup.
14. **ScoreMix**, [[Arxiv'2022](https://arxiv.org/pdf/2202.07570.pdf)]
   ScoreNet: Learning Non-Uniform Attention and Augmentation for Transformer-Based Histopathological Image Classification.
15. **RecursiveMix**, [[Arxiv'2022](https://arxiv.org/pdf/2203.06844.pdf)] [[code](https://github.com/implus/RecursiveMix-pytorch)]
   RecursiveMix: Mixed Learning with History.

## Label Mixup Methods

1. **MixUp**, [[ICLR'2018](https://arxiv.org/abs/1710.09412)] [[code](https://github.com/facebookresearch/mixup-cifar10)]
   mixup: Beyond Empirical Risk Minimization.
2. **CutMix**, [[ICCV'2019](https://arxiv.org/abs/1905.04899)] [[code](https://github.com/clovaai/CutMix-PyTorch)]
   CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.
3. **MetaMixup**, [[TNNLS'2021](https://arxiv.org/abs/1908.10059)]
   Metamixup: Learning adaptive interpolation policy of mixup with metalearning.
4. **mWH**, [[Arxiv'2021](https://arxiv.org/abs/2101.04342)] [[code](https://github.com/yuhao318/mwh)]
   Mixup Without Hesitation.
5. **CAMixup**, [[ICLR'2021](https://arxiv.org/abs/2010.09875)] [[code](https://github.com/google/edward2/tree/main/experimental/marginalization_mixup)]
   Combining Ensembles and Data Augmentation can Harm your Calibration.
6. **Saliency Grafting**, [[AAAI'2022](https://arxiv.org/abs/2112.08796)]
   Saliency Grafting: Innocuous Attribution-Guided Mixup with Calibrated Label Mixing.
7. **TransMix**, [[CVPR'2022](https://arxiv.org/pdf/2111.09833.pdf)] [[code](https://github.com/Beckschen/TransMix)]
   TransMix: Attend to Mix for Vision Transformers.
8. **DecoupleMix**, [[Arxiv'2022](https://arxiv.org/abs/2203.10761),] [[code](https://github.com/Westlake-AI/openmixup)]
   Decoupled Mixup for Data-efficient Learning.
9. **TokenMix**, [[ECCV'2022](https://arxiv.org/abs/2207.08409),] [[code](https://github.com/Sense-X/TokenMix)]
   TokenMix: Rethinking Image Mixing for Data Augmentation in Vision Transformers.

## Contribution

Feel free to send [pull requests](https://github.com/Westlake-AI/openmixup/pulls) to add more links! Current contributors include: Siyuan Li ([@Lupin1998](https://github.com/Lupin1998)) and Zicheng Liu ([@pone7](https://github.com/pone7)).
