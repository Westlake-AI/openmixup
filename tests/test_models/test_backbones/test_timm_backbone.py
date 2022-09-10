import pytest
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from openmixup.models.backbones import TIMMBackbone


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_timm_backbone():
    """Test timm backbones, features_only=False (default)."""
    with pytest.raises(TypeError):
        # TIMMBackbone has 1 required positional argument: 'model_name'
        model = TIMMBackbone(pretrained=True)

    with pytest.raises(TypeError):
        # pretrained must be bool
        model = TIMMBackbone(model_name='resnet18', pretrained='model.pth')

    # Test resnet18 from timm
    model = TIMMBackbone(model_name='resnet18')
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)
    assert isinstance(model.timm_model.global_pool.pool, nn.Identity)
    assert isinstance(model.timm_model.fc, nn.Identity)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size((1, 512, 7, 7))

    # Test efficientnet_b1 with pretrained weights
    model = TIMMBackbone(model_name='efficientnet_b1', pretrained=True)
    model.init_weights()
    model.train()
    assert isinstance(model.timm_model.global_pool.pool, nn.Identity)
    assert isinstance(model.timm_model.classifier, nn.Identity)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size((1, 1280, 7, 7))

    # Test vit_tiny_patch16_224 with pretrained weights
    model = TIMMBackbone(model_name='vit_tiny_patch16_224', pretrained=True)
    model.init_weights()
    model.train()
    assert isinstance(model.timm_model.head, nn.Identity)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    # Disable the test since TIMM's behavior changes between 0.5.4 and 0.5.5
    # assert feat[0].shape == torch.Size((1, 197, 192))


def test_timm_backbone_features_only():
    """Test timm backbones, features_only=True."""
    # Test different norm_layer, can be: 'SyncBN', 'BN2d', 'GN', 'LN', 'IN'
    # Test resnet18 from timm, norm_layer='BN2d'
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32,
        norm_layer='BN2d')

    # Test resnet18 from timm, norm_layer='SyncBN'
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32,
        norm_layer='SyncBN')

    # Test resnet18 from timm, output_stride=32
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)

