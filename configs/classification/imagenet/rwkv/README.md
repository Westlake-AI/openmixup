# RWKV

> [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)

## Abstract

Transformers have revolutionized almost all natural language processing (NLP) tasks but suffer from memory and computational complexity that scales quadratically with sequence length. In contrast, recurrent neural networks (RNNs) exhibit linear scaling in memory and computational requirements but struggle to match the same performance as Transformers due to limitations in parallelization and scalability. We propose a novel model architecture, Receptance Weighted Key Value (RWKV), that combines the efficient parallelizable training of Transformers with the efficient inference of RNNs. Our approach leverages a linear attention mechanism and allows us to formulate the model as either a Transformer or an RNN, which parallelizes computations during training and maintains constant computational and memory complexity during inference, leading to the first non-transformer architecture to be scaled to tens of billions of parameters. Our experiments reveal that RWKV performs on par with similarly sized Transformers, suggesting that future work can leverage this architecture to create more efficient models. This work presents a significant step towards reconciling the trade-offs between computational efficiency and model performance in sequence processing tasks. 

<div align=center>
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/44519745/249304190-fc9e0d1c-55c3-457a-8994-35bc8f36ecac.png" width="100%"/>
</div>

## Citation

```
@article{Peng2023RWKV,
  title={RWKV: Reinventing RNNs for the Transformer Era},
  author={Bo Peng and Eric Alcaide and Quentin G. Anthony and Alon Albalak and Samuel Arcadinho and Huanqi Cao and Xin Cheng and Michael Chung and Matteo Grella and G Kranthikiran and Xuzheng He and Haowen Hou and Przemyslaw Kazienko and Jan Kocoń and Jiaming Kong and Bartlomiej Koptyra and Hayden Lau and Krishna Sri Ipsit Mantri and Ferdinand Mom and Atsushi Saito and Xiangru Tang and Bolun Wang and Johan Sokrates Wind and Stansilaw Wozniak and Ruichong Zhang and Zhenyuan Zhang and Qihang Zhao and Peng Zhou and Jian Zhu and Rui Zhu},
  journal={ArXiv},
  year={2023},
  volume={abs/2305.13048}
}
```
