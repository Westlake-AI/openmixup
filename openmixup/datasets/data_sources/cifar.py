import os
from abc import ABCMeta, abstractmethod
from PIL import Image
import numpy as np
import torchvision

from ..registry import DATASOURCES


class Cifar(metaclass=ABCMeta):
    """ CIFAR-10 and CIFAR-100 datasets
        https://www.cs.toronto.edu/~kriz/cifar.html
    
    Args:
        root (str): Dataset root path, cotaining 'cifar-xxx-python'.
        split (str): Dataset split in ['train', 'test'].
        return_label (bool): Whether is Sup. or {Semi-Sup. & Self-Sup.}.
        num_labeled (int): If return_label==False, randomly select
            the num_labeled uniformly for each class.
    """

    CLASSES = None

    def __init__(self, root, split, return_label=True, num_labeled=None):
        assert split in ['train', 'test']
        self.root = root
        self.split = split
        self.return_label = return_label
        self.num_labeled = num_labeled
        self.cifar = None
        self.set_cifar()
        self.labels = self.cifar.targets
        self.set_split()

    @abstractmethod
    def set_cifar(self):
        pass

    @abstractmethod
    def set_split(self):
        pass

    def get_length(self):
        return len(self.cifar)

    def get_sample(self, idx):
        img = Image.fromarray(self.cifar.data[idx])
        if self.return_label:
            target = self.labels[idx]  # img: HWC, RGB
            return img, target
        else:
            return img


@DATASOURCES.register_module
class CIFAR10(Cifar):

    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck'
    ]

    def __init__(self, root, split, return_label=True, num_labeled=None):
        super().__init__(root, split, return_label, num_labeled)

    def set_cifar(self):
        try:
            self.cifar = torchvision.datasets.CIFAR10(
                root=self.root, train=self.split == 'train', download=False)
        except:
            raise Exception("Please download CIFAR10 manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")

    def set_split(self):
        """ set semi-supervised split (l or ul) """
        if self.split == 'test':
            return
        labeled_idx = None
        class_num = 10
        if self.return_label == True and self.num_labeled is not None:
            label_per_class = self.num_labeled // class_num
            labels = np.array(self.labels)
            labeled_idx = []
            # unlabeled: (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
            unlabeled_idx = np.array(range(len(labels)))
            for i in range(class_num):
                idx = np.where(labels == i)[0]
                idx = np.random.choice(idx, label_per_class, False)
                labeled_idx.extend(idx)
            labeled_idx = np.array(labeled_idx)
            assert len(labeled_idx) == self.num_labeled
            np.random.shuffle(labeled_idx)
        if labeled_idx is not None:
            self.cifar.data = self.cifar.data[labeled_idx]
            self.cifar.targets = labels[labeled_idx].tolist()
            self.labels = self.cifar.targets


@DATASOURCES.register_module
class CIFAR100(Cifar):

    SUPER_CLASSES = [
        'aquatic mammals', 'fish', 'flowers', 'food containers',
        'fruit and vegetables', 'household electrical devices',
        'household furniture', 'insects', 'large carnivores',
        'large man-made outdoor things', 'large natural outdoor scenes',
        'large omnivores and herbivores', 'medium-sized mammals',
        'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
        'trees', 'vehicles 1', 'vehicles 2'
    ]
    CLASSES = [
        'beaver', 'dolphin', 'otter', 'seal', 'whale',
        'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
        'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
        'bottles', 'bowls', 'cans', 'cups', 'plates',
        'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
        'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
        'bed', 'chair', 'couch', 'table', 'wardrobe',
        'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
        'bear', 'leopard', 'lion', 'tiger', 'wolf',
        'bridge', 'castle', 'house', 'road', 'skyscraper',
        'cloud', 'forest', 'mountain', 'plain', 'sea',
        'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
        'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
        'crab', 'lobster', 'snail', 'spider', 'worm',
        'baby', 'boy', 'girl', 'man', 'woman',
        'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
        'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
        'maple', 'oak', 'palm', 'pine', 'willow',
        'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
        'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
    ]

    def __init__(self, root, split, return_label=True, num_labeled=None):
        super().__init__(root, split, return_label, num_labeled)

    def set_cifar(self):
        try:
            self.cifar = torchvision.datasets.CIFAR100(
                root=self.root, train=self.split == 'train', download=False)
        except:
            raise Exception("Please download CIFAR10 manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")

    def set_split(self):
        """ set semi-supervised split (l or ul) """
        if self.split == 'test':
            return
        labeled_idx = None
        class_num = 100
        if self.return_label == True and self.num_labeled is not None:
            label_per_class = self.num_labeled // class_num
            labels = np.array(self.labels)
            labeled_idx = []
            # unlabeled: (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
            unlabeled_idx = np.array(range(len(labels)))
            for i in range(class_num):
                idx = np.where(labels == i)[0]
                idx = np.random.choice(idx, label_per_class, False)
                labeled_idx.extend(idx)
            labeled_idx = np.array(labeled_idx)
            assert len(labeled_idx) == self.num_labeled
            np.random.shuffle(labeled_idx)
        if labeled_idx is not None:
            self.cifar.data = self.cifar.data[labeled_idx]
            self.cifar.targets = labels[labeled_idx].tolist()
            self.labels = self.cifar.targets


class CIFAR_Corruption(metaclass=ABCMeta):

    def __init__(self, root):
        self.root = root
        self.data = None
        self.targets = None
        self.corruption_list = [
            "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
            "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
            "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
            "snow", "spatter", "speckle_noise", "zoom_blur"]
        self.set_cifar_corruption()
    
    def set_cifar_corruption(self):
        self.targets = list()
        self.data = list()
        # load labels
        targets = np.load(os.path.join(self.root, "labels.npy"))
        assert targets.shape[0] == 50000

        # load data
        for name in self.corruption_list:
            self.data.append(
                np.load(os.path.join(self.root, name+".npy"))
            )
            self.targets.append(targets)
        
        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        assert self.data.shape[0] == self.targets.shape[0]


@DATASOURCES.register_module
class CIFAR_C(Cifar):
    """ CIFAR-10 and CIFAR-100 Corruption 
    
    Implementation of "Benchmarking Neural Network Robustness to Common
    Corruptions and Perturbations (https://arxiv.org/pdf/1903.12261v1.pdf)".

    CIFAR-10 dataset download (https://zenodo.org/record/2535967).
    CIFAR-100 dataset download (https://zenodo.org/record/3555552).
    """

    def __init__(self, root, split, return_label=True):
        super().__init__(root, split, return_label)

    def set_cifar(self):
        assert self.split == 'test'
        try:
            self.cifar = CIFAR_Corruption(root=self.root)
        except:
            raise Exception("Data or label files are invalid, please check \
                whether the dataset is downloading from the official link.")

    def get_length(self):
        return self.cifar.targets.shape[0]
