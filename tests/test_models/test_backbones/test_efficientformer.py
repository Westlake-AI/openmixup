import pytest

from openmixup.models.backbones import EfficientFormer


def test_assertion():
    with pytest.raises(AssertionError):
        EfficientFormer(arch='unknown')

    with pytest.raises(AssertionError):
        # EfficientFormer arch dict should include 'embed_dims',
        EfficientFormer(arch=dict(embed_dims=[48, 96, 224, 448]))

    with pytest.raises(AssertionError):
        # EfficientFormer arch dict should include 'embed_dims',
        EfficientFormer(
            arch=dict(
                layers=[3, 2, 6, 4],
                embed_dims=[48, 96, 224, 448]
                downsamples=[False, True, True, True],
                vit_num=1)
            )


def test_efficientformer():

    # Test forward
    model = EfficientFormer(arch='l1', out_indices=-1)
    model.init_weights()
    model.train()
