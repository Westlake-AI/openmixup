import torch
import torch.nn as nn
import math
from ..registry import HEADS
from mmcv.cnn import NonLocal2d, kaiming_init, normal_init
from ..necks import ConvNeck
from .. import builder
from ..utils import build_norm_layer
from openmixup.utils import print_log


@HEADS.register_module
class PixelMixBlock_V2(nn.Module):
    """Pixel-wise MixBlock V2.
        version v08.24
            add pre_attn and pre_conv
        version v10.09
            add learnable lam mult
    
    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
            Default: True.
        double_norm (bool): Whether to scale pairwise_weight again by L1 norm.
            Default: False
        attention_mode (str): Options (non-local) are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
        unsampling_mode (str): Unsampling mode {'nearest', 'bilinear', etc}.
            Default: 'nearest'.
        pre_norm_cfg (dict): Config dict for a norm before q,k,v input of MixBlock.
            e.g., pre_norm_cfg=dict(type='BN', requires_grad=True).
            Default: None.
        pre_conv_cfg (dict): Config dict for a before MixBlock convolution neck.
            e.g., pre_conv_cfg=dict(
                type="ConvNeck", in_channels=256, hid_channels=128, out_channels=256,
                num_layers=2, kernel_size=3, with_bias=True, with_residual=True).
            Default: None.
        pre_attn_cfg (dict): Config dict for a before MixBlock self-attention block.
            e.g., pre_attn_cfg=dict(in_channels=256, mode='gaussian').
            Default: None.
        pre_neck_cfg (dict): Config dict for a Neck parallel to MixBlock, which converts
            feature maps to flattened vectors for the pre_head (directly supervised by loss).
                E.g., pre_neck_cfg=dict(
                    type='LinearNeck', in_channels=256, out_channels=128, with_avg_pool=True)
            Default: None.
        pre_head_cfg (dict): Config dict for a loss head parallel to MixBlock, e.g., infoNCE
            or classification CE loss, which is used to train pre_conv and pre_attn.
            Default: None.
        lam_concat (bool): Whether to concat lam as a channel in all input q, k, v.
            Default: False. (lam_concat=False if lam_concat_v=True)
        lam_concat_v (bool): Whether to concat lam as a channel in v but not in q, k.
            Default: False. (lam_concat_v=False if lam_concat=True)
        lam_mul (bool or float): Whether to mult lam in x_lam and mult (1-lam) in x_lam_
            to get pair-wise weight.
            Default: False.
        lam_mul_k (float): Rescale lambda before multipling to x, which is adjusted by k.
            Default: -1.
        lam_residual (bool): Whether to use residual addition for lam_mult.
            Default: False.
        value_neck_cfg (dict): Config dict for a non-linear value embedding network.
            E.g., value_neck_cfg=dict(
                type="ConvNeck", in_channels=256, hid_channels=128, out_channels=1, act_cfg=dict(type='ELU'),
                num_layers=2, kernel_size=1, with_bias=True, with_residual=False).
            Default: None. (default value network is 1x1 conv)
        x_qk_concat (bool): Whether to concat x and x_ in q, k pair-wise weight embedding.
            Default: False.
        x_v_concat (bool): Whether to concat x and x_ in value embedding.
            Default: False.
        mask_loss_mode (str): Which mode in {'L1', 'L2', 'none', 'Variance'} to caculate loss.
            Default: "none".
        mask_loss_margin (int): Margine loss for the grid mask pattens. Default: 0.
        mask_mode (str): Which mode to normalize mixup masks to sum=1. Default: "none".
    """
    
    def __init__(self,
            in_channels,
            reduction=2,
            use_scale=True,
            double_norm=False,
            attention_mode='embedded_gaussian',
            unsampling_mode='bilinear',
            pre_norm_cfg=None,
            pre_conv_cfg=None,
            pre_attn_cfg=None,
            pre_neck_cfg=None,
            pre_head_cfg=None,
            lam_concat=False,
            lam_concat_v=False,
            lam_mul=0.,
            lam_mul_k=-1,
            lam_residual=False,
            value_neck_cfg=None,
            x_qk_concat=False,
            x_v_concat=False,
            mask_loss_mode="none",
            mask_loss_margin=0,
            mask_mode="none",
            frozen=False):
        super(PixelMixBlock_V2, self).__init__()
        # non-local args
        self.in_channels = int(in_channels)
        self.reduction = int(reduction)
        self.use_scale = bool(use_scale)
        self.double_norm = bool(double_norm)
        self.inter_channels = max(in_channels // reduction, 1)
        self.attention_mode = str(attention_mode)
        self.unsampling_mode = str(unsampling_mode)
        assert self.attention_mode in ['gaussian', 'embedded_gaussian']
        assert self.unsampling_mode in [
            'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear',
        ]
        
        # pre MixBlock or parallel to MixBlock
        assert pre_norm_cfg is None or isinstance(pre_norm_cfg, dict)
        assert pre_conv_cfg is None or isinstance(pre_conv_cfg, dict)
        assert pre_attn_cfg is None or isinstance(pre_attn_cfg, dict)
        assert pre_neck_cfg is None or isinstance(pre_neck_cfg, dict)
        assert pre_head_cfg is None or isinstance(pre_head_cfg, dict)
        self.pre_norm = pre_norm_cfg
        self.pre_conv = pre_conv_cfg
        self.pre_attn = pre_attn_cfg
        self.pre_neck = pre_neck_cfg
        self.pre_head = pre_head_cfg
        if pre_norm_cfg is not None:
            _, self.pre_norm = build_norm_layer(pre_norm_cfg, in_channels)
        if pre_conv_cfg is not None:
            self.pre_conv = ConvNeck(**pre_conv_cfg)
        if pre_attn_cfg is not None:
            self.pre_attn = NonLocal2d(**pre_attn_cfg)
        if pre_neck_cfg is not None:
            self.pre_neck = builder.build_neck(pre_neck_cfg)
        if pre_head_cfg is not None:
            self.pre_head = builder.build_head(pre_head_cfg)

        # mixblock args
        self.lam_concat = bool(lam_concat)
        self.lam_concat_v = bool(lam_concat_v)
        self.lam_mul = float(lam_mul) if float(lam_mul) > 0 else 0
        self.lam_mul_k = float(lam_mul_k) if float(lam_mul_k) > 0 else -1
        self.lam_residual = bool(lam_residual)
        assert value_neck_cfg is None or isinstance(value_neck_cfg, dict)
        self.value_neck_cfg = value_neck_cfg
        self.x_qk_concat = bool(x_qk_concat)
        self.x_v_concat = bool(x_v_concat)
        self.mask_loss_mode = str(mask_loss_mode)
        self.mask_loss_margin = max(mask_loss_margin, 0.)
        self.mask_mode = str(mask_mode)
        self.frozen = bool(frozen)
        assert 0 <= lam_mul and lam_mul <= 1
        assert lam_mul_k == -1 or (lam_mul_k <= 10 and lam_mul_k >= 0)
        assert mask_loss_mode in [
            "none", "L2", "L1", "Variance", "L1+Variance", "L2+Variance", "Sparsity"]
        assert mask_mode in [
            "none", "none_v_", "sum", "softmax"]
        if self.lam_concat or self.lam_concat_v:
            assert self.lam_concat != self.lam_concat_v, \
                "lam_concat can be adopted on q,k,v or only on v"
        if self.lam_concat or self.lam_mul:
            assert self.lam_concat != self.lam_mul, \
                "both lam_concat and lam_mul change q,k,v in terms of lam"
        if self.lam_concat or self.x_qk_concat:
            assert self.lam_concat != self.x_qk_concat, \
                "x_lam=x_lam_=cat(x,x_) if x_qk_concat=True, it's no use to concat lam"

        # concat all as k,q,v
        self.qk_in_channels = int(in_channels + 1) \
            if self.lam_concat else int(in_channels)
        self.v_in_channels = int(in_channels + 1) \
            if self.lam_concat or self.lam_concat_v else int(in_channels)
        if self.x_qk_concat:
            self.qk_in_channels = int(2 * self.in_channels)
        if self.x_v_concat:
            self.v_in_channels = int(2 * self.in_channels)
        
        # MixBlock, conv value
        if value_neck_cfg is None:
            self.value = nn.Conv2d(
                self.v_in_channels,
                1,
                kernel_size=1,
                stride=1)
        else:
            value_neck_cfg["in_channels"] = self.v_in_channels
            self.value = builder.build_neck(value_neck_cfg)
        # MixBlock, conv q,k
        if self.attention_mode == 'embedded_gaussian':
            self.key = None
            if self.x_qk_concat:  # sym conv q and k
                # conv key
                self.key = nn.Conv2d(
                    self.qk_in_channels,
                    self.inter_channels,
                    kernel_size=1,
                    stride=1)
            # conv query
            self.query = nn.Conv2d(
                self.qk_in_channels,
                self.inter_channels,
                kernel_size=1,
                stride=1)
        
        self.init_weights()
        if self.frozen:
            self._freeze()

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        # init mixblock
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _freeze(self):
        # before mixblock
        if self.pre_norm is not None:
            self.pre_norm.eval()
        if self.pre_conv is not None:
            self.pre_conv.eval()
        if self.pre_attn is not None:
            self.pre_attn.eval()
        if self.pre_neck is not None:
            self.pre_neck.eval()
        if self.pre_head is not None:
            self.pre_head.eval()
        # mixblock
        self.value.eval()
        if self.attention_mode == 'embedded_gaussian':
            self.query.eval()
            if self.key is not None:
                self.key.eval()
        # detach
        if self.frozen:
            # before mixblock
            if self.pre_norm is not None:
                for param in self.pre_norm.parameters():
                    param.requires_grad = False
            if self.pre_conv is not None:
                for param in self.pre_conv.parameters():
                    param.requires_grad = False
            if self.pre_attn is not None:
                for param in self.pre_attn.parameters():
                    param.requires_grad = False
            if self.pre_neck is not None:
                for param in self.pre_neck.parameters():
                    param.requires_grad = False
            if self.pre_head is not None:
                for param in self.pre_head.parameters():
                    param.requires_grad = False
            # mixblock
            if self.attention_mode == 'embedded_gaussian':
                for param in self.query.parameters():
                    param.requires_grad = False
                if self.key is not None:
                    for param in self.key.parameters():
                        param.requires_grad = False
            for param in self.value.parameters():
                param.requires_grad = False

    def gaussian(self, q_x, k_x):
        """ non-local similarity func """
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(q_x, k_x)
        if torch.any(torch.isnan(pairwise_weight)):
            print_log("Warming attention map is nan, P: {}".format(pairwise_weight), logger='root')
            raise ValueError
        if self.use_scale:
            pairwise_weight /= q_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight
    
    def embedded_gaussian(self, q_x, k_x):
        """ learnable non-local similarity func """
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(q_x, k_x)
        if torch.any(torch.isnan(pairwise_weight)):
            print_log("Warming attention map is nan, P: {}".format(pairwise_weight), logger='root')
            raise ValueError
        if self.use_scale:
            # q_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= q_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        if self.double_norm:
            pairwise_weight = pairwise_weight / (1e-8 + pairwise_weight.sum(dim=1, keepdim=True))
        return pairwise_weight
    
    def rescale_lam_mult(self, lam, k=1):
        """ adjust lam against y=x in terms of k """
        assert k >= 0
        k += 1
        if not isinstance(lam, float):
            lam = float(lam)
        return 1 / (k - 2/3) * (4/3 * math.pow(lam, 3) -2 * lam**2 + k * lam)

    def forward(self, x, lam, index, scale_factor, debug=False):
        """ v08.23, add pre_conv and pre_attn

            x (tensor): Input feature map [N, C, H, W].
            lam (int): Mixup ratio lambda.
            index (tensor): Random shuffle index in current mini-batch.
            scale_factor (int): Unsampling factor (assert scale_factor % 2 == 0).
            debug (bool): Whether to use debug mode.
        """
        results = dict()
        # pre-step 0: input 2d feature map x, [N, C, H, W]
        if isinstance(x, list) and index is None:
            assert len(x) == 2  # only for SSL mixup
            x = torch.cat(x)
        n, _, h, w = x.size()
        # pre-step 1: before mixblock, add pre conv and attn
        if self.pre_attn is not None:
            x = self.pre_attn(x)
        if self.pre_conv is not None:
            x = self.pre_conv([x])[0]
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        if index is None:  # only for SSL mixup, [2N, C, H, W]
            n = n // 2
            x_lam  = x[:n, ...]
            x_lam_ = x[n:, ...]
        else:  # supervised cls
            x_lam  = x
            x_lam_ = x[index, :]  # shuffle within a gpu
        results = dict(x_lam=x_lam, x_lam_=x_lam_)
        
        # pre-step 2: lambda encoding
        if self.lam_mul > 0:  # multiply lam to x_lam
            assert self.lam_concat == False
            # rescale lam
            if self.lam_mul_k >= 0:
                lam_rescale = self.rescale_lam_mult(lam, self.lam_mul_k)
            else:
                lam_rescale = lam
            # using residual
            if self.lam_residual:
                x_lam = x_lam * (1 + lam_rescale * self.lam_mul)
                x_lam_ = x_lam_ * (1 + (1 - lam_rescale) * self.lam_mul)
            else:
                x_lam = x_lam * lam_rescale
                x_lam_ = x_lam_ * (1 - lam_rescale)
        if self.lam_concat:  # concat lam as a new channel
            # assert self.lam_mul > 0 and self.x_qk_concat == False
            lam_block = torch.zeros(n, 1, h, w).cuda()
            lam_block[:] = lam
            x_lam  = torch.cat([x_lam, lam_block], dim=1)
            x_lam_ = torch.cat([x_lam_, 1-lam_block], dim=1)
        
        # step 1: conpute 1x1 conv value, v: [N, HxW, 1].
        v, v_ = x_lam, x_lam_
        if self.x_v_concat:
            v  = torch.cat([x_lam, x_lam_], dim=1)
            v_ = v
        if self.lam_concat_v:
            lam_block = torch.zeros(n, 1, h, w).cuda()
            lam_block[:] = lam
            v  = torch.cat([x_lam, lam_block], dim=1)
            v_ = torch.cat([x_lam_, 1-lam_block], dim=1)
        if self.mask_mode != "none":  # compute both v and v_
            if self.value_neck_cfg is None:
                v_ = self.value(v_).view(n, 1, -1)  # [N, 1, HxW]
            else:
                v_ = self.value([v_])[0].view(n, 1, -1)  # [N, 1, HxW]
            v_ = v_.permute(0, 2, 1)  # v_ for 1-lam: [N, HxW, 1]
        if self.value_neck_cfg is None:
            v = self.value(v).view(n, 1, -1)  # [N, 1, HxW]
        else:
            v = self.value([v])[0].view(n, 1, -1)  # [N, 1, HxW]
        v = v.permute(0, 2, 1)  # v for lam: [N, HxW, 1]
        # debug mode
        if debug:
            debug_plot = dict(value=v_.view(n, h, -1).clone().detach())
        
        # step 2: compute 1x1 conv q & k, q_x: [N, HxW, C], k_x: [N, C, HxW].
        if self.x_qk_concat:
            x_lam = torch.cat([x_lam, x_lam_], dim=1)
            x_lam_ = x_lam
        if self.attention_mode == 'gaussian':
            q_x = x_lam.view(n, self.qk_in_channels, -1)
            q_x = q_x.permute(0, 2, 1)  # q for lam: [N, HxW, C]
            k_x = x_lam_.view(n, self.qk_in_channels, -1)  # k for 1-lam: [N, C, HxW]
        else:
            # query
            q_x = self.query(x_lam).view(n, self.inter_channels, -1)
            q_x = q_x.permute(0, 2, 1)  # q for lam: [N, HxW, C/r]
            # key
            if self.key is not None:
                k_x = self.key(x_lam_).view(n, self.inter_channels, -1)  # [N, C/r, HxW]
            else:
                k_x = self.query(x_lam_).view(n, self.inter_channels, -1)  # [N, C/r, HxW]

        # ste 3: 2d pairwise_weight: [N, HxW, HxW]
        pairwise_func = getattr(self, self.attention_mode)
        pairwise_weight = pairwise_func(q_x, k_x)  # x_lam [N, HxW, C/r] x [N, C/r, HxW] x_lam_

        # debug mode
        if debug:
            debug_plot["pairwise_weight"] = pairwise_weight.clone().detach()
            results["debug_plot"] = debug_plot
        
        # step 4: generate mask and upsampling
        mask_lam = torch.matmul(  # P^T x v_lam = mask_lam
            pairwise_weight.permute(0, 2, 1), v).view(n, 1, h, w)  # mask for lam
        if torch.any(torch.isnan(mask_lam)):
            print_log("Warming mask_lam is nan, P: {}, v: {}".format(pairwise_weight, v), logger='root')
            pairwise_weight = pairwise_weight.clamp(min=-1e20, max=1e20)
            mask_lam = torch.matmul(  # P^T x v_lam = mask_lam
                pairwise_weight.permute(0, 2, 1), v.clamp(min=-1e20, max=1e20)
            ).view(n, 1, h, w)
        upsampling = nn.Upsample(scale_factor=scale_factor, mode=self.unsampling_mode)
        mask_lam = upsampling(mask_lam)
        mask_lam = torch.sigmoid(mask_lam)  # mask for lam, in [0, 1]

        if self.mask_mode != "none":
            # P x v_lam_ = mask_lam_
            mask_lam_ = torch.matmul(pairwise_weight, v_).view(n, 1, h, w)  # 1 - lam
            if torch.any(torch.isnan(mask_lam_)):
                print_log("Warming mask_lam_ is nan, P: {}, v: {}".format(pairwise_weight, v_), logger='root')
                mask_lam = torch.matmul(
                    pairwise_weight, v_.clamp(min=-1e20, max=1e20)
                ).view(n, 1, h, w)
            mask_lam_ = upsampling(mask_lam_)
            mask_lam_ = torch.sigmoid(mask_lam_)  # mask for 1-lam
            if self.mask_mode == "sum":
                # stop grad of one side [try]
                mask = torch.cat([mask_lam.clone().detach(), mask_lam_], dim=1)
                # sum to 1
                sum_masks = mask.sum(1, keepdim=True)
                mask /= sum_masks
            elif self.mask_mode == "softmax":
                # stop grad of one side [try]
                mask = torch.cat([mask_lam.clone().detach(), mask_lam_], dim=1)
                # sum to 1 by softmax
                mask = mask.softmax(dim=1)
            elif self.mask_mode == "none_v_":
                mask_lam = None
                mask = torch.cat([1 - mask_lam_, mask_lam_], dim=1)
            else:
                raise NotImplementedError
        else:
            mask = torch.cat([mask_lam, 1 - mask_lam], dim=1)
        
        results["mask"] = mask
        return results

    def mask_loss(self, mask, lam):
        """ loss for mixup masks """
        losses = dict()
        assert mask.dim() == 4
        n, k, h, w = mask.size()  # mixup mask [N, 2, H, W]
        if k > 1:  # the second mask has no grad!
            mask = mask[:, 1, :, :].unsqueeze(1)
        m_mean = mask.sum() / (n * h * w)  # mask mean in [0, 1]
        zero = torch.tensor(0.).cuda()

        if self.mask_loss_mode == "L1":  # [0, 1-m]
            losses['loss'] = torch.max(torch.abs(1 - m_mean - lam) - self.mask_loss_margin, zero).mean()
        elif self.mask_loss_mode == "L2":  # [0, 1-m^2]
            losses['loss'] = torch.max((1 - m_mean - lam) ** 2 - self.mask_loss_margin ** 2, zero).mean()
        elif self.mask_loss_mode == "Variance":  # [0, 0.5]
            losses['loss'] = -torch.max((torch.sum((mask - m_mean)**2) / (n * h * w)), zero)
        elif self.mask_loss_mode == "Sparsity":  # [0, 0.25-m]
            losses['loss'] = torch.max(torch.abs(mask * (mask - 1)).sum() / (n * h * w) - self.mask_loss_margin, zero)
        elif self.mask_loss_mode == "L1+Variance":  # [0, 1-m] + [0, 1]
            losses['loss'] = torch.max(torch.abs(1 - m_mean - lam) - self.mask_loss_margin, zero).mean() - \
                2 * torch.max((torch.sum((mask - m_mean)**2) / (n * h * w)), zero)
        elif self.mask_loss_mode == "L2+Variance":  # [0, 1-m^2] + [0, 1]
            losses['loss'] = torch.max((1 - m_mean - lam) ** 2 - self.mask_loss_margin ** 2, zero).mean() - \
                2 * torch.max((torch.sum((mask - m_mean)**2) / (n * h * w)), zero)
        else:
            raise NotImplementedError
        if torch.isnan(losses['loss']):
            print_log("Warming mask loss: {}, mask sum: {}".format(losses['loss'], mask), logger='root')
            losses['loss'] = None
            # raise ValueError
        return losses
