import random

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import kaiming_init, constant_init

from ..registry import BACKBONES
from .base_backbone import BaseBackbone
from ..utils import grad_batch_shuffle_ddp, grad_batch_unshuffle_ddp


class BasicBlock(nn.Module):
    """BasicBlock for Wide ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        stride (int): stride of the block. Default: 1.
        drop_rate (float): Dropout ratio in the residual block. Default: 0.
        activate_before_residual (bool): Since the first conv in WRN doesn't
            have bn-relu behind, we use the bn1 and relu1 in the block1 to
            make up the ``conv1-bn1-relu1`` structure. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 drop_rate=0.0,
                 activate_before_residual=False,
                 with_cp=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.dropout = nn.Dropout(float(drop_rate)) \
            if float(drop_rate) > 0 else nn.Identity()
        self.with_cp = with_cp
        self.equalInOut = (in_channels == out_channels)
        self.convShortcut = None
        if stride != 1 or not self.equalInOut:
            self.convShortcut = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1, stride=stride, padding=0, bias=True)
        self.activate_before_residual = activate_before_residual

    def forward(self, x):

        def _inner_forward(x):
            if not self.equalInOut and self.activate_before_residual == True:
                x = self.relu1(self.bn1(x))
                out = self.relu2(self.bn2(self.conv1(x)))
            else:
                out = self.relu1(self.bn1(x))
                out = self.relu2(self.bn2(self.conv1(out)))
            out = self.dropout(out)
            out = self.conv2(out)
            if self.equalInOut:
                out += x
            else:
                out += self.convShortcut(x)

            return out

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class NetworkBlock(nn.Module):
    """" Network Block (stage) in Wide ResNet """
    
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 block,
                 stride,
                 drop_rate=0.0,
                 activate_before_residual=False,
                 with_cp=False):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(int(num_layers)):
            layers.append(block(i == 0 and in_channels or out_channels, out_channels,
                                i == 0 and stride or 1, drop_rate,
                                activate_before_residual, with_cp))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


@BACKBONES.register_module()
class WideResNet(BaseBackbone):
    """Wide Residual Networks backbone.

    A PyTorch implement of : `Wide Residual Networks
    <https://arxiv.org/abs/1605.07146>`_

    Modified from the `official repo
    https://github.com/szagoruyko/wide-residual-networks`_

    Args:
        first_stride (int): Stride of the first 3x3 conv. Default: 1.
        in_channels (int): Number of input image channels. Default: 3.
        depth (int): Network depth, from {10, 28, 37}, total 3 stages.
        widen_factor (int): Width of each stage convolution block. Default: 2.
        drop_rate (float): Dropout ratio in residual blocks. Default: 0.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(2, )``.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 first_stride,
                 in_channels=3,
                 depth=28,
                 widen_factor=2,
                 drop_rate=0.0,
                 out_indices=(0, 1, 2,),
                 frozen_stages=-1,
                 norm_eval=False,
                 with_cp=False):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        
        # 1st conv before any network block, 3x3
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], BasicBlock, first_stride,
            drop_rate, activate_before_residual=True, with_cp=with_cp)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], BasicBlock, 2, drop_rate,
            activate_before_residual=False, with_cp=with_cp)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], BasicBlock, 2, drop_rate,
            activate_before_residual=False, with_cp=with_cp)
        # original: global average pooling and classifier (in head)
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.channels = channels[3]
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        
        self._freeze_stages()
    
    def init_weights(self, pretrained=None):
        super(WideResNet, self).init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_out', nonlinearity='leaky_relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                    constant_init(m, val=1, bias=0)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.conv1]:
                for param in m.parameters():
                    param.requires_grad = False
            for i in range(self.frozen_stages + 1):
                m = getattr(self, 'block{}'.format(i+1))
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
            if self.frozen_stages == 2:
                for m in [self.bn1]:
                    for param in m.parameters():
                        param.requires_grad = False

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        for i in range(3):
            block_i = getattr(self, 'block{}'.format(i+1))
            x = block_i(x)
            if i == 2:  # after block3
                x = self.relu(self.bn1(x))
                # x = F.adaptive_avg_pool2d(x, 1)
                # x = x.view(-1, self.channels)  # Nxd
            if i in self.out_indices:
                outs.append(x)
                if len(self.out_indices) == 1:
                    return outs
        return outs

    def train(self, mode=True):
        super(WideResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()


@BACKBONES.register_module()
class WideResNet_Mix(WideResNet):
    """Wide-ResNet Support ManifoldMix and its variants
        v12.10

    Provide a port to mixup the latent space.
    """

    def __init__(self, **kwargs):
        super(WideResNet_Mix, self).__init__(**kwargs)
    
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
            mix_layer = mix_args["layer"]  # {0, 1, 2,}
            if mix_args["BN_shuffle"]:
                x, _, idx_unshuffle = grad_batch_shuffle_ddp(x)  # 2N index if cross_view
            else:
                idx_unshuffle = None
        else:
            mix_layer = -1
        
        # input mixup
        if mix_layer == 0:
            x = self._feature_mixup(x, idx_unshuffle_BN=idx_unshuffle, **mix_args)
        # normal conv1
        x = self.conv1(x)

        outs = []
        # block 1 to 3
        for i in range(3):
            block_i = getattr(self, 'block{}'.format(i+1))
            x = block_i(x)
            if i == 2:  # after block3
                x = self.relu(self.bn1(x))
                # x = F.adaptive_avg_pool2d(x, 1)
                # x = x.view(-1, self.channels)  # Nxd
            if i in self.out_indices:
                outs.append(x)
                if len(self.out_indices) == 1:
                    return outs
            if i+1 == mix_layer:
                x = self._feature_mixup(x, idx_unshuffle_BN=idx_unshuffle, **mix_args)
        return outs
