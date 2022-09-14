import pytest

from openmixup.models.backbones import HorNet


def test_assertion():
    with pytest.raises(AssertionError):
        HorNet(arch='unknown')

    with pytest.raises(AssertionError):
        # HorNet arch dict should include 'embed_dims',
        HorNet(arch=dict(base_dim=64))

    with pytest.raises(AssertionError):
        # HorNet arch dict should include 'embed_dims',
        HorNet(
            arch=dict(
                base_dim=64,
                depths=[2, 3, 18, 2],
                orders=[2, 3, 4, 5],
                dw_cfg=[dict(type='DW', kernel_size=7)] * 4,
            )
        )


def test_hornet():

    # Test forward
    model = HorNet(arch='tiny', out_indices=-1)
    model.init_weights()
    model.train()
