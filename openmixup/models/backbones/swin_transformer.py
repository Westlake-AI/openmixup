# reference: https://github.com/open-mmlab/mmclassification/tree/master/mmcls/models/backbones
# modified from mmclassification swin_transformer.py
from copy import deepcopy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed, PatchMerging
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init, trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils.parrots_wrapper import _BatchNorm

from openmixup.utils import get_root_logger
from ..utils import (ShiftWindowMSA, to_2tuple, resize_pos_embed,
                     resize_relative_position_bias_table,
                     grad_batch_shuffle_ddp, grad_batch_unshuffle_ddp)  # for mixup
from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class SwinBlock(BaseModule):
    """Swin Transformer block.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        shift (bool): Shift the attention window or not. Defaults to False.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_path (float): The drop path rate after attention and ffn.
            Defaults to 0.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        feat_scale (bool): If True, use FeatScale (anti-oversmoothing).
            FeatScale re-weights feature maps on separate frequency bands
            to amplify the high-frequency signals.
            Defaults to False.
        attn_scale (bool): If True, use AttnScale (anti-oversmoothing).
            AttnScale decomposes a self-attention block into low-pass and
            high-pass components, then rescales and combines these two filters
            to produce an all-pass self-attention matrix.
            Defaults to False.
        attn_cfgs (dict): The extra config of Shift Window-MSA.
            Defaults to empty dict.
        ffn_cfgs (dict): The extra config of FFN. Defaults to empty dict.
        norm_cfg (dict): The config of norm layers.
            Defaults to ``dict(type='LN')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size=7,
                 shift=False,
                 ffn_ratio=4.,
                 drop_path=0.,
                 pad_small_map=False,
                 feat_scale=False,
                 attn_scale=False,
                 attn_cfgs=dict(),
                 ffn_cfgs=dict(),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__(init_cfg)
        self.with_cp = with_cp

        _attn_cfgs = {
            'embed_dims': embed_dims,
            'num_heads': num_heads,
            'shift_size': window_size // 2 if shift else 0,
            'window_size': window_size,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'pad_small_map': pad_small_map,
            'attn_scale': attn_scale,
            **attn_cfgs
        }
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(**_attn_cfgs)

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': 0,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': dict(type='GELU'),
            **ffn_cfgs
        }
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(**_ffn_cfgs)

        self.feat_scale = feat_scale
        if self.feat_scale:
            self.lamb1 = nn.Parameter(
                torch.zeros(embed_dims), requires_grad=True)
            self.lamb2 = nn.Parameter(
                torch.zeros(embed_dims), requires_grad=True)

    def freq_scale(self, x):
        if not self.feat_scale:
            return x
        x_d = torch.mean(x, -2, keepdim=True)  # [bs, 1, dim]
        x_h = x - x_d  # high freq [bs, len, dim]
        x_d = x_d * self.lamb1
        x_h = x_h * self.lamb2
        x = x + x_d + x_h
        return x

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = self.freq_scale(x)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlockSequence(BaseModule):
    """Module with successive Swin Transformer blocks and downsample layer.

    Args:
        embed_dims (int): Number of input channels.
        depth (int): Number of successive swin transformer blocks.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window. Defaults to 7.
        downsample (bool): Downsample the output of blocks by patch merging.
            Defaults to False.
        downsample_cfg (dict): The extra config of the patch merging layer.
            Defaults to empty dict.
        drop_paths (Sequence[float] | float): The drop path rate in each block.
            Defaults to 0.
        block_cfgs (Sequence[dict] | dict): The extra config of each block.
            Defaults to empty dicts.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        feat_scale (bool): If True, use FeatScale (anti-oversmoothing).
            FeatScale re-weights feature maps on separate frequency bands
            to amplify the high-frequency signals.
            Defaults to False.
        attn_scale (bool): If True, use AttnScale (anti-oversmoothing).
            AttnScale decomposes a self-attention block into low-pass and
            high-pass components, then rescales and combines these two filters
            to produce an all-pass self-attention matrix.
            Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 depth,
                 num_heads,
                 window_size=7,
                 downsample=False,
                 downsample_cfg=dict(),
                 drop_paths=0.,
                 block_cfgs=dict(),
                 with_cp=False,
                 pad_small_map=False,
                 feat_scale=False,
                 attn_scale=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth

        if not isinstance(block_cfgs, Sequence):
            block_cfgs = [deepcopy(block_cfgs) for _ in range(depth)]

        self.embed_dims = embed_dims
        self.blocks = ModuleList()
        for i in range(depth):
            _block_cfg = {
                'embed_dims': embed_dims,
                'num_heads': num_heads,
                'window_size': window_size,
                'shift': False if i % 2 == 0 else True,
                'drop_path': drop_paths[i],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                'feat_scale': feat_scale,
                'attn_scale': attn_scale,
                **block_cfgs[i]
            }
            block = SwinBlock(**_block_cfg)
            self.blocks.append(block)

        if downsample:
            _downsample_cfg = {
                'in_channels': embed_dims,
                'out_channels': 2 * embed_dims,
                'norm_cfg': dict(type='LN'),
                **downsample_cfg
            }
            self.downsample = PatchMerging(**_downsample_cfg)
        else:
            self.downsample = None

    def forward(self, x, in_shape):
        for block in self.blocks:
            x = block(x, in_shape)

        if self.downsample:
            x, out_shape = self.downsample(x, in_shape)
        else:
            out_shape = in_shape
        return x, out_shape

    @property
    def out_channels(self):
        if self.downsample:
            return self.downsample.out_channels
        else:
            return self.embed_dims


@BACKBONES.register_module()
class SwinTransformer(BaseBackbone):
    """Swin Transformer.

    A PyTorch implement of : `Swin Transformer: Hierarchical Vision Transformer
    using Shifted Windows <https://arxiv.org/abs/2103.14030>`_

    Modified from the `official repo
    <https://github.com/microsoft/Swin-Transformer>`

    Args:
        arch (str | dict): Swin Transformer architecture. If use string, choose
            from 'tiny', 'small', 'base' and 'large'. If use dict, it should
            have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **num_heads** (List[int]): The number of heads in attention
              modules of each stage.

            Defaults to 'tiny'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 4.
        in_channels (int): The num of input channels. Defaults to 3.
        window_size (int): The height and width of the window. Defaults to 7.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults to False.
        interpolate_mode (str): Select the interpolate mode for absolute
            position embeding vector resize. Defaults to "bicubic".
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        feat_scale (bool): If True, use FeatScale (anti-oversmoothing).
            FeatScale re-weights feature maps on separate frequency bands
            to amplify the high-frequency signals.
            Defaults to False.
        attn_scale (bool): If True, use AttnScale (anti-oversmoothing).
            AttnScale decomposes a self-attention block into low-pass and
            high-pass components, then rescales and combines these two filters
            to produce an all-pass self-attention matrix.
            Defaults to False.
        norm_cfg (dict): Config dict for normalization layer for all output
            features. Defaults to ``dict(type='LN')``
        stage_cfgs (Sequence[dict] | dict): Extra config dict for each
            stage. Defaults to an empty dict.
        patch_cfg (dict): Extra config dict for patch embedding.
            Defaults to an empty dict.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from mmcls.models import SwinTransformer
        >>> import torch
        >>> extra_config = dict(
        >>>     arch='tiny',
        >>>     stage_cfgs=dict(downsample_cfg={'kernel_size': 3,
        >>>                                     'expansion_ratio': 3}))
        >>> self = SwinTransformer(**extra_config)
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> output = self.forward(inputs)
        >>> print(output.shape)
        (1, 2592, 4)
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
<<<<<<< HEAD
                         'depths': [2, 2, 6, 2],
                         'num_heads': [3, 6, 12, 24]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [3, 6, 12, 24]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths': [2, 2, 18, 2],
                         'num_heads': [6, 12, 24, 48]}),
=======
                         'depths':     [2, 2,  6,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [6, 12, 24, 48]}),
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    }  # yapf: disable

    _version = 3
    num_extra_tokens = 0

    def __init__(self,
                 arch='tiny',
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 window_size=7,
                 drop_rate=0.,
                 drop_path_rate=0.1,
<<<<<<< HEAD
                 out_indices=(3,),
=======
                 out_indices=(3, ),
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
                 use_abs_pos_embed=False,
                 interpolate_mode='bicubic',
                 with_cp=False,
                 frozen_stages=-1,
                 norm_eval=False,
                 pad_small_map=False,
                 feat_scale=False,
                 attn_scale=False,
                 norm_cfg=dict(type='LN'),
                 stage_cfgs=dict(),
                 patch_cfg=dict(),
                 init_cfg=None):
        super(SwinTransformer, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_heads'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.num_layers = len(self.depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.interpolate_mode = interpolate_mode
        self.frozen_stages = frozen_stages

        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=dict(type='LN'),
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
<<<<<<< HEAD

=======
        
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        if self.use_abs_pos_embed:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, self.embed_dims))
            self._register_load_state_dict_pre_hook(
                self._prepare_abs_pos_embed)

        self._register_load_state_dict_pre_hook(
            self._prepare_relative_position_bias_table)

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.norm_eval = norm_eval

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.stages = ModuleList()
        embed_dims = [self.embed_dims]
        for i, (depth,
                num_heads) in enumerate(zip(self.depths, self.num_heads)):
            if isinstance(stage_cfgs, Sequence):
                stage_cfg = stage_cfgs[i]
            else:
                stage_cfg = deepcopy(stage_cfgs)
            downsample = True if i < self.num_layers - 1 else False
            _stage_cfg = {
                'embed_dims': embed_dims[-1],
                'depth': depth,
                'num_heads': num_heads,
                'window_size': window_size,
                'downsample': downsample,
                'drop_paths': dpr[:depth],
                'with_cp': with_cp,
                'pad_small_map': pad_small_map,
                'feat_scale': feat_scale,
                'attn_scale': attn_scale,
                **stage_cfg
            }

            stage = SwinBlockSequence(**_stage_cfg)
            self.stages.append(stage)

            dpr = dpr[depth:]
            embed_dims.append(stage.out_channels)

        for i in self.out_indices:
            if i < 0:
                continue
            if norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg, embed_dims[i + 1])[1]
            else:
                norm_layer = nn.Identity()

            self.add_module(f'norm{i}', norm_layer)

    def init_weights(self, pretrained=None):
        super(SwinTransformer, self).init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is None:
                for m in self.modules():
                    if isinstance(m, (nn.Linear)):
                        trunc_normal_init(m, std=0.02)
                    elif isinstance(m, (
<<<<<<< HEAD
                            nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
=======
                        nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
                        constant_init(m, val=1, bias=0)
            # pos_embed & cls_token
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.absolute_pos_embed, self.patch_resolution, hw_shape,
                self.interpolate_mode, self.num_extra_tokens)
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               stage.out_channels).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args,
                              **kwargs):
        """load checkpoints."""
        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None
<<<<<<< HEAD
            or version < 2) and self.__class__ is SwinTransformer:
=======
                or version < 2) and self.__class__ is SwinTransformer:
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
            final_stage_num = len(self.stages) - 1
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if k.startswith('norm.') or k.startswith('backbone.norm.'):
                    convert_key = k.replace('norm.', f'norm{final_stage_num}.')
                    state_dict[convert_key] = state_dict[k]
                    del state_dict[k]
        if (version is None
<<<<<<< HEAD
            or version < 3) and self.__class__ is SwinTransformer:
=======
                or version < 3) and self.__class__ is SwinTransformer:
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                if 'attn_mask' in k:
                    del state_dict[k]

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      *args, **kwargs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(0, self.frozen_stages + 1):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        for i in self.out_indices:
            if i < 0:
                continue
            if i <= self.frozen_stages:
                for param in getattr(self, f'norm{i}').parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                    m.eval()

    def _prepare_abs_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'absolute_pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.absolute_pos_embed.shape != ckpt_pos_embed_shape:
            logger = get_root_logger()
            logger.info(
                'Resize the absolute_pos_embed shape from '
                f'{ckpt_pos_embed_shape} to {self.absolute_pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    def _prepare_relative_position_bias_table(self, state_dict, prefix, *args,
                                              **kwargs):
        state_dict_model = self.state_dict()
        all_keys = list(state_dict_model.keys())
        for key in all_keys:
            if 'relative_position_bias_table' in key:
                ckpt_key = prefix + key
                if ckpt_key not in state_dict:
                    continue
                relative_position_bias_table_pretrained = state_dict[ckpt_key]
                relative_position_bias_table_current = state_dict_model[key]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if L1 != L2:
<<<<<<< HEAD
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)
=======
                    src_size = int(L1**0.5)
                    dst_size = int(L2**0.5)
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
                    new_rel_pos_bias = resize_relative_position_bias_table(
                        src_size, dst_size,
                        relative_position_bias_table_pretrained, nH1)
                    logger = get_root_logger()
                    logger.info('Resize the relative_position_bias_table from '
                                f'{state_dict[ckpt_key].shape} to '
                                f'{new_rel_pos_bias.shape}')
                    state_dict[ckpt_key] = new_rel_pos_bias

                    # The index buffer need to be re-generated.
                    index_buffer = ckpt_key.replace('bias_table', 'index')
                    del state_dict[index_buffer]


@BACKBONES.register_module()
class SwinTransformer_Mix(SwinTransformer):
    """Swin Transformer.

    Provide a port to mixup the latent space for both SL and SSL.
    """

    def __init__(self, **kwargs):
        super(SwinTransformer_Mix, self).__init__(**kwargs)

    def _feature_mixup(self, x, mask, dist_shuffle=False, idx_shuffle_mix=None,
                       cross_view=False, BN_shuffle=False, idx_shuffle_BN=None,
                       idx_unshuffle_BN=None, **kwargs):
        """ mixup two feature maps with the pixel-wise mask
<<<<<<< HEAD

=======
        
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
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
<<<<<<< HEAD
        if dist_shuffle == True:  # cross gpus shuffle
=======
        if dist_shuffle==True:  # cross gpus shuffle
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
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
        if 2 * x.size(3) == mask[0].size(3):
            mask[0] = nn.functional.interpolate(mask[0], scale_factor=0.5, mode="nearest")
            mask[1] = nn.functional.interpolate(mask[1], scale_factor=0.5, mode="nearest")
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
        bs = x.size(0)

        # input mixup
        if mix_layer == 0:
            x = self._feature_mixup(x, idx_unshuffle_BN=idx_unshuffle, **mix_args)

        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.absolute_pos_embed, self.patch_resolution, hw_shape,
                self.interpolate_mode, self.num_extra_tokens)
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               stage.out_channels).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
<<<<<<< HEAD
            if i + 1 == mix_layer:  # stage 1 to 4
=======
            if i+1 == mix_layer:  # stage 1 to 4
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
                x = x.view(bs, *hw_shape, -1).permute(0, 3, 1, 2).contiguous()
                x = self._feature_mixup(x, idx_unshuffle_BN=idx_unshuffle, **mix_args)
                x = x.flatten(2).transpose(1, 2)

<<<<<<< HEAD
        return outs
=======
        return outs
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
