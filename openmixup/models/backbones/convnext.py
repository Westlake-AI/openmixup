from functools import partial
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn import ModuleList, Sequential

from mmcv.cnn.bricks import build_activation_layer, build_norm_layer, DropPath
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from ..utils import (GRN, LayerNorm2d, lecun_normal_init,
                     grad_batch_shuffle_ddp, grad_batch_unshuffle_ddp)  # for mixup


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        dw_conv_cfg (dict): Config of depthwise convolution.
            Defaults to ``dict(kernel_size=7, padding=3)``.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = build_activation_layer(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        if use_grn:
            self.grn = GRN(mid_channels)
        else:
            self.grn = None

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                x = self.pointwise_conv1(x)
                x = self.act(x)

                if self.grn is not None:
                    x = self.grn(x, data_format='channel_first')
                x = self.pointwise_conv2(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@BACKBONES.register_module()
class ConvNeXt(BaseBackbone):
    """ConvNeXt v1&v2 backbone.

    A PyTorch implementation of : `A ConvNet for the 2020s
    <https://arxiv.org/pdf/2201.03545.pdf>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    To use ConvNeXt v2, please set ``use_grn=True`` and ``layer_scale_init_value=0.``.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        use_grn (bool): Whether to add Global Response Normalization in the
            blocks. Defaults to False.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict.
    """  # noqa: E501
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 norm_eval=False,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ],
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0])[1],
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1])[1],
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices and i == 3:
                norm_layer = build_norm_layer(norm_cfg, channels)[1]
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def init_weights(self, pretrained=None):
        super(ConvNeXt, self).init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is not None:
                return
            for m in self.modules():
                if isinstance(m, (nn.Conv2d)):
                    lecun_normal_init(m, mode='fan_in', distribution='truncated_normal')
                elif isinstance(m, (nn.Linear)):
                    trunc_normal_init(m, mean=0., std=0.02, bias=0)
                elif isinstance(m, (
                    _BatchNorm, nn.LayerNorm, LayerNorm2d, GRN, nn.GroupNorm)):
                    constant_init(m, val=1, bias=0)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False
    
    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                if i == 3:
                    norm_layer = getattr(self, f'norm{i}')
                    if self.gap_before_final_norm and i == 3:
                        gap = x.mean([-2, -1], keepdim=True)
                        x = norm_layer(gap).flatten(1)
                    else:
                        x = norm_layer(x)
                outs.append(x)
                if len(self.out_indices) == 1:
                    return outs

        return outs

    def train(self, mode=True):
        super(ConvNeXt, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                    m.eval()


@BACKBONES.register_module()
class ConvNeXt_Mix(ConvNeXt):
    """ConvNeXt.

    Provide a port to mixup the latent space for both SL and SSL.
    """

    def __init__(self, **kwargs):
        super(ConvNeXt_Mix, self).__init__(**kwargs)

    def _feature_mixup(self, x, mask, dist_shuffle=False, idx_shuffle_mix=None,
                       cross_view=False, BN_shuffle=False, idx_shuffle_BN=None,
                       idx_unshuffle_BN=None, **kwargs):
        """ mixup two feature maps with the pixel-wise mask

        Args:
            x, mask (tensor): Input x [N,C,H,W] and mixup mask [N, \*, H, W].
            dist_shuffle (bool): Whether to shuffle cross gpus.
            idx_shuffle_mix (tensor): Shuffle indice of [N,1] to generate x_.
            cross_view (bool): Whether to view the input x as two views [2N, C, H, W],
                which is usually adopted in self-supervised and semi-supervised settings.
            BN_shuffle (bool): Whether to do shuffle cross gpus for shuffle_BN.
            idx_shuffle_BN (tensor): Shuffle indice to utilize shuffle_BN cross gpus.
            idx_unshuffle_BN (tensor): Unshuffle indice for the shuffle_BN (in pair).
        """
        # adjust mixup mask
        assert mask.dim() == 4 and mask.size(1) <= 2
        if mask.size(1) == 1:
            mask = [mask, 1 - mask]
        else:
            mask = [
                mask[:, 0, :, :].unsqueeze(1), mask[:, 1, :, :].unsqueeze(1)]
        # undo shuffle_BN for ssl mixup
        if BN_shuffle:
            assert idx_unshuffle_BN is not None and idx_shuffle_BN is not None
            x = grad_batch_unshuffle_ddp(x, idx_unshuffle_BN)  # 2N index if cross_view

        # shuffle input
        if dist_shuffle==True:  # cross gpus shuffle
            assert idx_shuffle_mix is not None
            if cross_view:
                N = x.size(0) // 2
                x_ = x[N:, ...].clone().detach()
                x = x[:N, ...]
                x_, _, _ = grad_batch_shuffle_ddp(x_, idx_shuffle_mix)
            else:
                x_, _, _ = grad_batch_shuffle_ddp(x, idx_shuffle_mix)
        else:  # within each gpu
            if cross_view:
                # default: the input image is shuffled
                N = x.size(0) // 2
                x_ = x[N:, ...].clone().detach()
                x = x[:N, ...]
            else:
                x_ = x[idx_shuffle_mix, :]
        assert x.size(3) == mask[0].size(3), \
            "mismatching mask x={}, mask={}.".format(x.size(), mask[0].size())
        mix = x * mask[0] + x_ * mask[1]

        # redo shuffle_BN for ssl mixup
        if BN_shuffle:
            mix, _, _ = grad_batch_shuffle_ddp(mix, idx_shuffle_BN)  # N index

        return mix

    def forward(self, x, mix_args=None):
        """ only support mask-based mixup policy """
        # latent space mixup
        if mix_args is not None:
            assert isinstance(mix_args, dict)
            mix_layer = mix_args["layer"]  # {0, 1, 2, 3}
            if mix_args["BN_shuffle"]:
                x, _, idx_unshuffle = grad_batch_shuffle_ddp(x)  # 2N index if cross_view
            else:
                idx_unshuffle = None
        else:
            mix_layer = -1

        # input mixup
        if mix_layer == 0:
            x = self._feature_mixup(x, idx_unshuffle_BN=idx_unshuffle, **mix_args)

        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                if i == 3:
                    norm_layer = getattr(self, f'norm{i}')
                    if self.gap_before_final_norm and i == 3:
                        gap = x.mean([-2, -1], keepdim=True)
                        x = norm_layer(gap).flatten(1)
                    else:
                        x = norm_layer(x)
                outs.append(x)
                if len(self.out_indices) == 1:
                    return outs
            if i+1 == mix_layer:  # stage 1 to 4
                x = self._feature_mixup(x, idx_unshuffle_BN=idx_unshuffle, **mix_args)

        return outs


@BACKBONES.register_module()
class MIMConvNeXt(ConvNeXt):
    """ConvNeXt backbone for MIM pre-training.

    Args:
        mask_layer (int): Layer to start MIM (mask img and add mask_token).
            Defaults to 0.
        mask_token (str): Mode of applying mask token in {None, 'randn', 'zero',
            'learnable', 'mean'}. Defaults to 'learnable'.
        mask_init (float): The init values of mask_token gamma. Defaults to 0.0.
    """

    def __init__(self,
                 mask_layer=0,
                 mask_token='learnable',
                 mask_init=0,
                 replace=True,
                 detach=False,
                 **kwargs):
        super(MIMConvNeXt, self).__init__(**kwargs)
        self.mask_layer = mask_layer
        self.mask_mode = mask_token
        self.replace = replace
        self.detach = detach
        assert self.mask_layer in [0, 1, 2, 3]
        assert self.mask_mode in [
            None, 'randn', 'zero', 'mean', 'learnable']
        self.mask_dims = self.channels[self.mask_layer]
        if self.mask_mode is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, self.mask_dims, 1, 1))
        if mask_init > 0 and not replace:
            self.mask_gamma = nn.Parameter(
                mask_init * torch.ones((1, self.mask_dims, 1, 1)), requires_grad=True)
        else:
            self.mask_gamma = None

    def init_weights(self, pretrained=None):
        """Initialize weights."""
        super(MIMConvNeXt, self).init_weights(pretrained)

        if pretrained is None:
            if self.mask_mode is not None:
                if self.mask_mode != 'zero':
                    trunc_normal_init(self.mask_token, mean=0., std=0.02, bias=0)
                if self.mask_mode != 'learnable':
                    self.mask_token.requires_grad = False

    def forward_mask(self, x, mask=None):
        """ perform MIM with mask and mask_token """
        B, _, H, W = x.size()
        if self.mask_mode is None:
            return x
        else:
            if self.mask_mode == 'mean':
                self.mask_token.data = x.mean(dim=[0, 2, 3], keepdim=True)
            mask_token = self.mask_token.expand(B, -1, H, W)
        assert mask is not None
        mask = mask.view(B, 1, H, W).type_as(mask_token)
        if self.replace:
            x = x * (1. - mask) + mask_token * mask
        else:
            if self.detach:
                x = x * (1. - mask) + x.clone().detach() * mask
            if self.mask_gamma is not None:
                x = x * (1. - mask) + (x * mask) * self.mask_gamma
            x = x + mask_token * mask  # residual
        return x

    def forward(self, x, mask=None):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)

            if self.mask_layer == i:  # mask, add mask token
                x = self.forward_mask(x, mask)
            x = stage(x)

            if i in self.out_indices:
                if i == 3:
                    norm_layer = getattr(self, f'norm{i}')
                    if self.gap_before_final_norm and i == 3:
                        gap = x.mean([-2, -1], keepdim=True)
                        x = norm_layer(gap).flatten(1)
                    else:
                        x = norm_layer(x)
                outs.append(x)
                if len(self.out_indices) == 1:
                    return outs

        return outs


@BACKBONES.register_module()
class ConvNeXt_CIFAR(ConvNeXt):
    """ConvNeXt backbone for CIFAR.

    Compared to standard ConvNeXt, it uses `kernel_size=3` and `stride=1` in
    the stem, which is more efficient than standard ConvNeXt on CIFAR.

    A PyTorch implementation of : `A ConvNet for the 2020s
    <https://arxiv.org/pdf/2201.03545.pdf>`_
    """

    def __init__(self, in_channels=3, norm_cfg=dict(type='LN2d', eps=1e-6), **kwargs):
        super(ConvNeXt_CIFAR, self).__init__(
            in_channels=in_channels, norm_cfg=norm_cfg, **kwargs)

        # the first stem layer
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=3,
                stride=1,
                padding=1),
            build_norm_layer(norm_cfg, self.channels[0])[1],
        )
        self.downsample_layers[0] = stem


@BACKBONES.register_module()
class ConvNeXt_Mix_CIFAR(ConvNeXt):
    """ConvNeXt backbone for CIFAR, support ManifoldMix and its variants

    Provide a port to mixup the latent space.
    """
    def __init__(self, in_channels=3, norm_cfg=dict(type='LN2d', eps=1e-6), **kwargs):
        super(ConvNeXt_Mix_CIFAR, self).__init__(
            in_channels=in_channels, norm_cfg=norm_cfg, **kwargs)

        # the first stem layer
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=3,
                stride=1,
                padding=1),
            build_norm_layer(norm_cfg, self.channels[0])[1],
        )
        self.downsample_layers[0] = stem

    def _feature_mixup(self, x, mask, dist_shuffle=False, idx_shuffle_mix=None,
                       cross_view=False, BN_shuffle=False, idx_shuffle_BN=None,
                       idx_unshuffle_BN=None, **kwargs):
        """ mixup two feature maps with the pixel-wise mask

        Args:
            x, mask (tensor): Input x [N,C,H,W] and mixup mask [N, \*, H, W].
            dist_shuffle (bool): Whether to shuffle cross gpus.
            idx_shuffle_mix (tensor): Shuffle indice of [N,1] to generate x_.
            cross_view (bool): Whether to view the input x as two views [2N, C, H, W],
                which is usually adopted in self-supervised and semi-supervised settings.
            BN_shuffle (bool): Whether to do shuffle cross gpus for shuffle_BN.
            idx_shuffle_BN (tensor): Shuffle indice to utilize shuffle_BN cross gpus.
            idx_unshuffle_BN (tensor): Unshuffle indice for the shuffle_BN (in pair).
        """
        # adjust mixup mask
        assert mask.dim() == 4 and mask.size(1) <= 2
        if mask.size(1) == 1:
            mask = [mask, 1 - mask]
        else:
            mask = [
                mask[:, 0, :, :].unsqueeze(1), mask[:, 1, :, :].unsqueeze(1)]
        # undo shuffle_BN for ssl mixup
        if BN_shuffle:
            assert idx_unshuffle_BN is not None and idx_shuffle_BN is not None
            x = grad_batch_unshuffle_ddp(x, idx_unshuffle_BN)  # 2N index if cross_view
        
        # shuffle input
        if dist_shuffle==True:  # cross gpus shuffle
            assert idx_shuffle_mix is not None
            if cross_view:
                N = x.size(0) // 2
                detach_p = random.random()
                x_ = x[N:, ...].clone().detach() if detach_p < 0.5 else x[N:, ...]
                x = x[:N, ...] if detach_p < 0.5 else x[:N, ...].detach()
                x_, _, _ = grad_batch_shuffle_ddp(x_, idx_shuffle_mix)
            else:
                x_, _, _ = grad_batch_shuffle_ddp(x, idx_shuffle_mix)
        else:  # within each gpu
            if cross_view:
                # default: the input image is shuffled
                N = x.size(0) // 2
                detach_p = random.random()
                x_ = x[N:, ...].clone().detach() if detach_p < 0.5 else x[N:, ...]
                x = x[:N, ...] if detach_p < 0.5 else x[:N, ...].detach()
            else:
                x_ = x[idx_shuffle_mix, :]
        assert x.size(3) == mask[0].size(3), \
            "mismatching mask x={}, mask={}.".format(x.size(), mask[0].size())
        mix = x * mask[0] + x_ * mask[1]

        # redo shuffle_BN for ssl mixup
        if BN_shuffle:
            mix, _, _ = grad_batch_shuffle_ddp(mix, idx_shuffle_BN)  # N index
        
        return mix

    def forward(self, x, mix_args=None):
        """ only support mask-based mixup policy """
        # latent space mixup
        if mix_args is not None:
            assert isinstance(mix_args, dict)
            mix_layer = mix_args["layer"]  # {0, 1, 2, 3}
            if mix_args["BN_shuffle"]:
                x, _, idx_unshuffle = grad_batch_shuffle_ddp(x)  # 2N index if cross_view
            else:
                idx_unshuffle = None
        else:
            mix_layer = -1
        
        # input mixup
        if mix_layer == 0:
            x = self._feature_mixup(x, idx_unshuffle_BN=idx_unshuffle, **mix_args)

        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                if i == 3:
                    norm_layer = getattr(self, f'norm{i}')
                    if self.gap_before_final_norm and i == 3:
                        gap = x.mean([-2, -1], keepdim=True)
                        x = norm_layer(gap).flatten(1)
                    else:
                        x = norm_layer(x)
                outs.append(x)
                if len(self.out_indices) == 1:
                    return outs
            if i+1 == mix_layer:
                x = self._feature_mixup(x, idx_unshuffle_BN=idx_unshuffle, **mix_args)
        return outs
