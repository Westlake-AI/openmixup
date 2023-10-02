from abc import ABCMeta, abstractmethod
from PIL import Image

import os
import random
import torch
import torchvision
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
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


def get_all_batches(loader):
    data_iter = iter(loader)
    steplen = len(loader)
    img_list, label_list = [], []

    for step in range(steplen):
        images, labels = data_iter.next()
        img_list.append(images)
        label_list.append(labels)
    img_tensor = torch.cat(img_list, dim=0)
    label_tensor = torch.cat(label_list, dim=0)
    
    return img_tensor, label_tensor


def rotate_img(img, degree=None, background=None):
    rotate_class = [(360.0 / 60) * i for i in range(60)]
    if degree is None:
        degree = random.sample(rotate_class, 1)[0]

    # rotate a image by PIL
    if background is not None:
        # img = img_torch2numpy(img) * 255
        img = np.transpose(img.numpy(), (1, 2, 0)) * 255
        pil_img = Image.fromarray(np.uint8(img))
        r_img = pil_img.rotate(degree)
        r_img = np.array(r_img)
        background[r_img > 2] = 255
        background = torch.from_numpy(background / 255).type(torch.float32)
        return background, degree
    else:
        # img = img / 2 + 0.5  # unnormalize
        pil_img = transforms.ToPILImage()(img)
        r_img = pil_img.rotate(degree)
        r_img = transforms.ToTensor()(r_img)  # read float
        # r_img = (r_img - 0.5) * 2.0  # normalize
        return r_img, degree


def get_rotate_imgs(imgs, background=None):
    # rotate set of images
    r_img_list, degree_list = [], []
    for i in range(imgs.shape[0]):
        b_img = None if background is None else background[i % background.shape[0]]
        r_img, degree = rotate_img(imgs[i], background=b_img)
        r_img_list.append(r_img.unsqueeze(0))
        degree_list.append(torch.Tensor([degree]))

    r_img_list = torch.cat(r_img_list, dim=0)
    degree_list = torch.cat(degree_list, dim=0)
    r_img_np = r_img_list.numpy()
    degree_np = degree_list.numpy()

    return r_img_np, degree_np


@DATASOURCES.register_module
class RCFMNIST(object):

    CLASSES = None

    def __init__(self, root, split, return_label=True):
        assert split in ['train', 'test']
        self.root = root
        self.split = split
        self.return_label = return_label
        self.set_mnist()

    def set_mnist(self):
        try:
            basic_transforms = transforms.Compose([
                transforms.Pad(2),
                transforms.Grayscale(3),
                transforms.ToTensor(),
            ])
            _split = self.split == 'train'
            mnist = torchvision.datasets.FashionMNIST(
                root=os.path.join(self.root, 'fmnist'), train=_split,
                download=True, transform=basic_transforms)
            cifar = torchvision.datasets.CIFAR10(
                root=os.path.join(self.root, 'cifar10'), train=_split,
                download=True, transform=None)
            data_loader = DataLoader(mnist, batch_size=1000, shuffle=False, num_workers=2)
            data_raw, _ = get_all_batches(data_loader)
            self.data, self.labels = get_rotate_imgs(data_raw, background=cifar.data)
            self.data = (self.data * 255).astype(np.uint8)
        except:
            raise Exception("Please download FashionMNIST and CIFAR10 manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")

    def get_length(self):
        return len(self.data)

    def get_sample(self, idx):
        img = self.data[idx]
        # return a PIL Image for transform in pipelines
        img = Image.fromarray(img)
        if self.return_label:
            target = int(self.labels[idx])  # img: HWC, RGB
            return img, target
        else:
            return img
