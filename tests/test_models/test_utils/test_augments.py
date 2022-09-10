from pyexpat import features
import pytest
import torch

from openmixup.models.utils import mixup, cutmix, smoothmix, saliencymix, attentivemix, fmix, puzzlemix, gridmix, resizemix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

images = torch.randn(4, 3, 32, 32)    # images obeying normal distribution with size [4, 3, 32, 32]
images = images.to(device)
labels = torch.randint(0, 10, (4, 1)) # labels obeying normal distribution with size [4, 4]

    
def test_mixup():   # test MixUp
    mixed_imgs, mixed_labels = mixup(images, labels)
    assert mixed_imgs[0].shape == torch.Size([3, 32, 32])
    assert mixed_labels[0].shape == torch.Size([4, 1])
    
def test_cutmix():  # test CutMix
    mixed_imgs, mixed_labels = cutmix(images, labels)
    assert mixed_imgs[0].shape == torch.Size([3, 32, 32])
    assert mixed_labels[0].shape == torch.Size([4, 1])

def test_smoothmix():    # test SmoothMix
    mixed_imgs, mixed_labels = smoothmix(images, labels)
    assert mixed_imgs[0].shape == torch.Size([3, 32, 32])
    assert mixed_labels[0].shape == torch.Size([4, 1])
    
def test_saliencymix():    # test SaliencyMix
    mixed_imgs, mixed_labels = saliencymix(images, labels)
    assert mixed_imgs[0].shape == torch.Size([3, 32, 32])
    assert mixed_labels[0].shape == torch.Size([4, 1])

def test_attentivemix():    # test AttentiveMix
    features_attentive = torch.randn(4, 3, 32, 32)    
    features_attentive = features_attentive.to(device)
    mixed_imgs, mixed_labels = attentivemix(images, labels, features=features_attentive)
    assert mixed_imgs[0].shape == torch.Size([3, 32, 32])
    assert mixed_labels[0].shape == torch.Size([4, 1])
    
def test_fmix():    # test FMix
    mixed_imgs, mixed_labels = fmix(images, labels)
    assert mixed_imgs[0].shape == torch.Size([3, 32, 32])
    assert mixed_labels[0].shape == torch.Size([4, 1])
    
def test_puzzlemix():    # test PuzzleMix
    features_puzzle = torch.randn(4, 32, 32)   
    features_puzzle = features_puzzle.to(device)
    mixed_imgs, mixed_labels = puzzlemix(images, labels, features=features_puzzle)
    assert mixed_imgs[0].shape == torch.Size([3, 32, 32])
    assert mixed_labels[0].shape == torch.Size([4, 1])

def test_gridmix():    # test GridMix
    mixed_imgs, mixed_labels = gridmix(images, labels, n_holes=4)
    assert mixed_imgs[0].shape == torch.Size([3, 32, 32])
    assert mixed_labels[0].shape == torch.Size([4, 1])

def test_resizemix():   # test ResizeMix
    mixed_imgs, mixed_labels = resizemix(images, labels)
    assert mixed_imgs[0].shape == torch.Size([3, 32, 32])
    assert mixed_labels[0].shape == torch.Size([4, 1])


