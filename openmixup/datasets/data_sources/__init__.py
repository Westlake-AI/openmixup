from .cifar import CIFAR10, CIFAR100, CIFAR_C
from .image_list import ImageList
from .imagenet import ImageNet
<<<<<<< HEAD
from .mnist import MNIST, FMNIST, KMNIST, USPS
from .palm_vein import Palm_Vein
from .cars import Cars
=======
from .mnist import MNIST, FMNIST, KMNIST, RCFMNIST, USPS
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)

__all__ = [
    'CIFAR10', 'CIFAR100', 'CIFAR_C',
    'ImageList', 'ImageNet',
<<<<<<< HEAD
    'MNIST', 'FMNIST', 'KMNIST', 'USPS',
    'Palm_Vein', 'Cars'
=======
    'MNIST', 'FMNIST', 'KMNIST', 'RCFMNIST', 'USPS',
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
]
