from abc import ABCMeta, abstractmethod
from PIL import Image

import torchvision

from ..registry import DATASOURCES


class Mnist_base(metaclass=ABCMeta):

    CLASSES = None

    def __init__(self, root, split, return_label=True):
        assert split in ['train', 'test']
        self.root = root
        self.split = split
        self.return_label = return_label
        self.mnist = None
        self.set_mnist()
        self.labels = self.mnist.targets

    @abstractmethod
    def set_mnist(self):
        pass

    def get_length(self):
        return len(self.mnist)

    def get_sample(self, idx):
        img = self.mnist.data[idx]
        # return a PIL Image for transform in pipelines
        img = Image.fromarray(img.numpy(), mode='L')
        if self.return_label:
            target = int(self.labels[idx])  # img: HWC, RGB
            return img, target
        else:
            return img


@DATASOURCES.register_module
class USPS(Mnist_base):

    CLASSES = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, split, return_label=True):
        super().__init__(root, split, return_label)
    
    def set_mnist(self):
        try:
            self.mnist = torchvision.datasets.USPS(
                root=self.root, train=self.split == 'train', download=False)
        except:
            raise Exception("Please download USPS binary manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")


@DATASOURCES.register_module
class MNIST(Mnist_base):

    CLASSES = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, split, return_label=True):
        super().__init__(root, split, return_label)
    
    def set_mnist(self):
        try:
            self.mnist = torchvision.datasets.MNIST(
                root=self.root, train=self.split == 'train', download=False)
        except:
            raise Exception("Please download MNIST manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")


@DATASOURCES.register_module
class FMNIST(Mnist_base):

    CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def __init__(self, root, split, return_label=True):
        super().__init__(root, split, return_label)
    
    def set_mnist(self):
        try:
            self.mnist = torchvision.datasets.FashionMNIST(
                root=self.root, train=self.split == 'train', download=False)
        except:
            raise Exception("Please download FashionMNIST manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")


@DATASOURCES.register_module
class KMNIST(Mnist_base):

    CLASSES = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']

    def __init__(self, root, split, return_label=True):
        super().__init__(root, split, return_label)
    
    def set_mnist(self):
        try:
            self.mnist = torchvision.datasets.KMNIST(
                root=self.root, train=self.split == 'train', download=False)
        except:
            raise Exception("Please download KMNIST manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")
