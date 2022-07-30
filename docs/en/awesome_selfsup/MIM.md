# Awesome Mask Image Modeling for Visual Represention

 [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![GitHub stars](https://img.shields.io/github/stars/Westlake-AI/openmixup?color=green) ![visitors](https://visitor-badge.glitch.me/badge?page_id=Westlake-AI/openmixup)

**We summarize mask image modeling (MIM) methods proposed for self-supervised visual representation learning.**
The list of awesome MIM methods is summarized in chronological order and is on updating.

## MIM for Backbones

### Fundermental Methods

1. **iGPT**, [[ICML'2020](http://proceedings.mlr.press/v119/chen20s/chen20s.pdf)] [[code](https://github.com/openai/image-gpt)]
   Generative Pretraining from Pixels.
2. **ViT**, [[ICLR'2021](https://arxiv.org/abs/2010.11929)] [[code](https://github.com/google-research/vision_transformer)]
   An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
3. **BEiT**, [[ICLR'2022](https://arxiv.org/abs/2106.08254)] [[code](https://github.com/microsoft/unilm/tree/master/beit)]
   BEiT: BERT Pre-Training of Image Transformers.
4. **MAE**, [[CVPR'2022](https://arxiv.org/abs/2111.06377)] [[code](https://github.com/facebookresearch/mae)]
   Masked Autoencoders Are Scalable Vision Learners.
5. **SimMIM**, [[CVPR'2022](https://arxiv.org/abs/2111.09886)] [[code](https://github.com/microsoft/simmim)]
   SimMIM: A Simple Framework for Masked Image Modeling.
6. **MaskFeat**, [[CVPR'2022](https://arxiv.org/abs/2112.09133)] [[code](https://github.com/facebookresearch/SlowFast)]
   Masked Feature Prediction for Self-Supervised Visual Pre-Training.
7. **SplitMask**, [[ArXiv'2021](https://arxiv.org/abs/2112.10740)] [None]
   Are Large-scale Datasets Necessary for Self-Supervised Pre-training?
8. **PeCo**, [[ArXiv'2021](https://arxiv.org/abs/2111.12710)] [[code](https://github.com/microsoft/PeCo)]
   PeCo: Perceptual Codebook for BERT Pre-training of Vision Transformers.
9. **MC-SSL0.0**, [[ArXiv'2021](https://arxiv.org/abs/2111.15340)] [None]
   MC-SSL0.0: Towards Multi-Concept Self-Supervised Learning.
10. **mc-BEiT**, [[ECCV'2022](https://arxiv.org/abs/2203.15371)] [[code](https://github.com/lixiaotong97/mc-BEiT)]
   mc-BEiT: Multi-choice Discretization for Image BERT Pre-training.
11. **BootMAE**, [[ECCV'2022](https://arxiv.org/abs/2207.07116)] [[code](https://github.com/LightDXY/BootMAE)]
   Bootstrapped Masked Autoencoders for Vision BERT Pretraining.
12. **SupMAE**, [[ArXiv'2022](https://arxiv.org/abs/2205.14540)] [[code](https://github.com/cmu-enyac/supmae)]
   SupMAE: Supervised Masked Autoencoders Are Efficient Vision Learners.
13. **MVP**, [[ArXiv'2022](https://arxiv.org/abs/2203.05175)] [None]
   MVP: Multimodality-guided Visual Pre-training.
14. **Ge2AE**, [[ArXiv'2022](https://arxiv.org/abs/2204.08227)] [None]
   The Devil is in the Frequency: Geminated Gestalt Autoencoder for Self-Supervised Visual Pre-Training.
15. **ConvMAE**, [[ArXiv'2022](https://arxiv.org/abs/2205.03892)] [[code](https://github.com/Alpha-VL/ConvMAE)]
   ConvMAE: Masked Convolution Meets Masked Autoencoders.
16. **GreenMIM**, [[ArXiv'2022](https://arxiv.org/abs/2205.13515)] [[code](https://github.com/LayneH/GreenMIM)]
   Green Hierarchical Vision Transformer for Masked Image Modeling.
17. **HiViT**, [[ArXiv'2022](https://arxiv.org/abs/2205.14949)] [None]
   HiViT: Hierarchical Vision Transformer Meets Masked Image Modeling.
18. **ObjMAE**, [[ArXiv'2022](https://arxiv.org/abs/2205.14338)] [None]
   Object-wise Masked Autoencoders for Fast Pre-training.
19. **LoMaR**, [[ArXiv'2022](https://arxiv.org/abs/2206.00790)] [[code](https://github.com/junchen14/LoMaR)]
   Efficient Self-supervised Vision Pretraining with Local Masked Reconstruction.

### MIM with Constrastive Learning

1. **MST**, [[NIPS'2021](https://arxiv.org/abs/2106.05656)] [None]
   MST: Masked Self-Supervised Transformer for Visual Representation.
2. **iBOT**, [[ICLR'2022](https://arxiv.org/abs/2111.07832)] [[code](https://github.com/bytedance/ibot)]
   iBOT: Image BERT Pre-Training with Online Tokenizer.
3. **MSN**, [[ArXiv'2022](https://arxiv.org/abs/2204.07141)] [[code](https://github.com/facebookresearch/msn)]
   Masked Siamese Networks for Label-Efficient Learning.
4. **SIM**, [[ArXiv'2022](https://arxiv.org/abs/2206.01204)] [[code](https://github.com/fundamentalvision/Siamese-Image-Modeling)]
   Siamese Image Modeling for Self-Supervised Vision Representation Learning.
5. **ConMIM**, [[ArXiv'2022](https://arxiv.org/abs/2205.09616)] [None]
   Masked Image Modeling with Denoising Contrast.
6. **RePre**, [[ArXiv'2022](https://arxiv.org/abs/2201.06857)] [None]
   RePre: Improving Self-Supervised Vision Transformer with Reconstructive Pre-training.

### MIM for Transformer and CNN

1. **Context-Encoder**, [[CVPR'2016](https://arxiv.org/abs/1604.07379)] [[code](https://github.com/pathak22/context-encoder)]
   Context Encoders: Feature Learning by Inpainting.
2. **CIM**, [[ArXiv'2022](https://arxiv.org/abs/2202.03382)] [None]
   Corrupted Image Modeling for Self-Supervised Visual Pre-Training.
3. **A2MIM**, [[ArXiv'2022](https://arxiv.org/abs/2205.13943)] [[code](https://github.com/Westlake-AI/openmixup)]
   Architecture-Agnostic Masked Image Modeling - From ViT back to CNN.
4. **MixMIM**, [[ArXiv'2022](https://arxiv.org/abs/2205.13137)] [[code](https://github.com/Sense-X/MixMIM)]
   MixMIM: Mixed and Masked Image Modeling for Efficient Visual Representation Learning.
5. **MRA**, [[ArXiv'2022](https://arxiv.org/abs/2206.04846)] [[code](https://github.com/haohang96/mra)]
   Masked Autoencoders are Robust Data Augmentors.

### MIM with Advanced Masking

1. **ADIOS**, [[ICML'2022](https://arxiv.org/abs/2201.13100)] [[code](https://github.com/YugeTen/adios)]
   Adversarial Masking for Self-Supervised Learning.
2. **AttMask**, [[ECCV'2022](https://arxiv.org/abs/2203.12719)] [[code](https://github.com/gkakogeorgiou/attmask)]
   What to Hide from Your Students: Attention-Guided Masked Image Modeling.
3. **UnMAE**, [[ArXiv'2022](https://arxiv.org/abs/2205.10063)] [[code](https://github.com/implus/um-mae)]
   Uniform Masking: Enabling MAE Pre-training for Pyramid-based Vision Transformers with Locality.
4. **SemMAE**, [[ArXiv'2022](https://arxiv.org/abs/2206.10207)] [None]
   SemMAE: Semantic-Guided Masking for Learning Masked Autoencoders.


## MIM in Downstream Tasks

### Object Detection

1. **MIMDet**, [[ArXiv'2022](https://arxiv.org/abs/2204.02964)] [[code](https://github.com/hustvl/MIMDet)]
   Unleashing Vanilla Vision Transformer with Masked Image Modeling for Object Detection.

### Video Rrepresentation

1. **VideoMAE**, [[ArXiv'2022](https://arxiv.org/abs/2203.12602)] [[code](https://github.com/MCG-NJU/VideoMAE)]
   VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.
2. **MAE**, [[ArXiv'2022](https://arxiv.org/abs/2205.09113)] [None]
   Masked Autoencoders As Spatiotemporal Learners.
3. **MaskViT**, [[ArXiv'2022](https://arxiv.org/abs/2206.11894)] [[code](https://github.com/agrimgupta92/maskvit)]
   MaskViT: Masked Visual Pre-Training for Video Prediction.
4. **MILES**, [[ArXiv'2022](https://arxiv.org/abs/2204.12408)] [[code](https://github.com/tencentarc/mcq)]
   MILES: Visual BERT Pre-training with Injected Language Semantics for Video-text Retrieval.

### Medical Image

1. **MedMAE**, [[ArXiv'2022](https://arxiv.org/abs/2203.05573)] [None]
   Self Pre-training with Masked Autoencoders for Medical Image Analysis.
2. **SD-MAE**, [[ArXiv'2022](https://arxiv.org/abs/2203.16983)] [None]
   Self-distillation Augmented Masked Autoencoders for Histopathological Image Classification.

### 3D Point Cloud

1. **PointBERT**, [[CVPR'2022](https://arxiv.org/abs/2111.14819)] [[code](https://github.com/lulutang0608/Point-BERT)]
   Pre-Training 3D Point Cloud Transformers with Masked Point Modeling.
2. **PointMAE**, [[ECCV'2022](https://arxiv.org/abs/2203.06604)] [[code](https://github.com/Pang-Yatian/Point-MAE)]
   Masked Autoencoders for Point Cloud Self-supervised Learning.


## Analysis of MIM

1. [[ICLR'2021](https://arxiv.org/abs/2006.05576)] [[code](https://github.com/yaohungt/Self_Supervised_Learning_Multiview)]
   Demystifying Self-Supervised Learning: An Information-Theoretical Framework.
2. [[ICLR'2021](https://arxiv.org/abs/2010.03648)] [None]
   A Mathematical Exploration of Why Language Models Help Solve Downstream Tasks.
3. [[NIPS'2021](https://arxiv.org/abs/2008.01064)] [None]
   Predicting What You Already Know Helps: Provable Self-Supervised Learning.
4. [[ArXiv'2022](https://arxiv.org/abs/2202.03670)] [None]
   How to Understand Masked Autoencoders.
5. [[ArXiv'2022](https://arxiv.org/abs/2202.09305)] [None]
   Masked prediction tasks: a parameter identifiability view.
6. [[ArXiv'2022](https://arxiv.org/abs/2205.13543)] [None]
   Revealing the Dark Secrets of Masked Image Modeling.
7. [[ArXiv'2022](https://arxiv.org/abs/2206.04664)] [None]
   On Data Scaling in Masked Image Modeling.
8. [[ArXiv'2022](https://arxiv.org/abs/2206.03826)] [None]
   Towards Understanding Why Mask-Reconstruction Pretraining Helps in Downstream Tasks.


## Contribution

Feel free to send [pull requests](https://github.com/Westlake-AI/openmixup/pulls) to add more links! Current contributors include: Siyuan Li ([@Lupin1998](https://github.com/Lupin1998)).
