import pytest
import torch

from openmixup.models.backbones import ConvNeXt


def test_assertion():
    with pytest.raises(AssertionError):
        ConvNeXt(arch='unknown')

    with pytest.raises(AssertionError):
        # ConvNeXt arch dict should include 'embed_dims',
        ConvNeXt(arch=dict(channels=[2, 3, 4, 5]))

    with pytest.raises(AssertionError):
        # ConvNeXt arch dict should include 'embed_dims',
        ConvNeXt(arch=dict(depths=[2, 3, 4], channels=[2, 3, 4, 5]))


def test_convnext():

    # Test forward
    model = ConvNeXt(arch='tiny', out_indices=-1)
    model.init_weights()
    model.train()
