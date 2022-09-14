import pytest

from openmixup.models.backbones import MViT


def test_assertion():
    with pytest.raises(AssertionError):
        MViT(arch='unknown')

    with pytest.raises(AssertionError):
        # MViT arch dict should include 'embed_dims',
        MViT(arch=dict(embeds=96))

    with pytest.raises(AssertionError):
        # MViT arch dict should include 'embed_dims',
        MViT(
            arch=dict(
                embeds=96,
                num_layers=10,
                num_heads=1,
                downscale_indices=[1, 3, 8])
            )


def test_mvit():

    # Test forward
    model = MViT(arch='tiny', out_indices=-1)
    model.init_weights()
    model.train()
