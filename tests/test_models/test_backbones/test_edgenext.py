import pytest

from openmixup.models.backbones import EdgeNeXt


def test_assertion():
    with pytest.raises(AssertionError):
        EdgeNeXt(arch='unknown')

    with pytest.raises(AssertionError):
        # EdgeNeXt arch dict should include 'embed_dims',
        EdgeNeXt(arch=dict(channels=[48, 96, 160, 304]))

    with pytest.raises(AssertionError):
        # EdgeNeXt arch dict should include 'embed_dims',
        EdgeNeXt(
            arch=dict(
                depths=[3, 3, 9, 3],
                channels=[48, 96, 160, 304],
                num_heads=[8, 8, 8, 8])
        )


def test_edgenext():

    # Test forward
    model = EdgeNeXt(arch='small', out_indices=-1)
    model.init_weights()
    model.train()
