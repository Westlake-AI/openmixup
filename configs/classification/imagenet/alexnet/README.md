# AlexNet

> [ImageNet classification with deep convolutional neural networks](https://dl.acm.org/doi/10.1145/3065386)

## Abstract

We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0%, respectively, which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overfitting in the fully connected layers we employed a recently developed regularization method called "dropout" that proved to be very effective. We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry.

<div align=center>
<img src="https://user-images.githubusercontent.com/44519745/204873304-0a481bc9-dbfc-4bb1-9139-5b499cff6ec4.png" width="90%"/>
</div>

## Results and models

We provide the implementation of AlexNet with PyTorch-style training setting.

### ImageNet-1k

| Model | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config |
|:---:|:---:|:---:|:---:|:---:|:---:|
| AlexNet | 61.1 | 0.72 | 62.5 | 83.0 | [config](./alexnet_4xb64_cos_ep100.py) |

## Citation

```
@article{2017Krizhevsky,
  author = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E.},
  title = {ImageNet Classification with Deep Convolutional Neural Networks},
  year = {2017},
  journal = {Commun. ACM},
  month = {may},
  pages = {84–90},
  numpages = {7}
}
```
