from copy import deepcopy
from unittest import TestCase

import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from openmixup.models.backbones import MlpMixer


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


class TestMLPMixer(TestCase):

    def setUp(self):
        self.cfg = dict(
            arch='b',
            img_size=224,
            patch_size=16,
            drop_rate=0.1,
            init_cfg=[
                dict(
                    type='Kaiming',
                    layer='Conv2d',
                    mode='fan_in',
                    nonlinearity='linear')
            ])

    def test_arch(self):
        # Test invalid default arch
        with self.assertRaisesRegex(AssertionError, 'not in default archs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = 'unknown'
            MlpMixer(**cfg)

        # Test invalid custom arch
        with self.assertRaisesRegex(AssertionError, 'Custom arch needs'):
            cfg = deepcopy(self.cfg)
            cfg['arch'] = {
                'embed_dims': 24,
                'num_layers': 16,
                'tokens_mlp_dims': 4096
            }
            MlpMixer(**cfg)

        # Test custom arch
        cfg = deepcopy(self.cfg)
        cfg['arch'] = {
            'embed_dims': 128,
            'num_layers': 6,
            'tokens_mlp_dims': 256,
            'channels_mlp_dims': 1024
        }
        model = MlpMixer(**cfg)
        self.assertEqual(model.embed_dims, 128)
        self.assertEqual(model.num_layers, 6)
        for layer in model.layers:
            self.assertEqual(layer.token_mix.feedforward_channels, 256)
            self.assertEqual(layer.channel_mix.feedforward_channels, 1024)

    def test_init_weights(self):
        # test weight init cfg
        cfg = deepcopy(self.cfg)
        cfg['init_cfg'] = [
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]
        model = MlpMixer(**cfg)
        ori_weight = model.patch_embed.projection.weight.clone().detach()
        model.init_weights()
        initialized_weight = model.patch_embed.projection.weight
        self.assertFalse(torch.allclose(ori_weight, initialized_weight))
