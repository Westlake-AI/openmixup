import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import BaseModule, force_fp32

from openmixup.utils import print_log
from ..registry import HEADS
from .. import builder


@HEADS.register_module
class AdaptiveMask(BaseModule):
    def __init__(self,
                 in_channel=256,
                 reduction=2,
                 lam_mul=False,
                 lam_mul_k=0.25,
                 lam_residual=True,
                 lam_concat=True,
                 use_scale=True,
                 unsampling_mode='nearest',
                 scale_factor=4,
                 att_norm_cfg=None,
                 frozen=False,
                 init_cfg=None,
                 **kwargs):
        super(AdaptiveMask, self).__init__(init_cfg)
        self.in_channel = int(in_channel)
        self.reduction = int(reduction)
        self.inter_channels = max(in_channel // reduction, 1)
        self.lam_mul = lam_mul
        self.lam_mul_k = [lam_mul_k] if isinstance(lam_mul_k, (int, float)) else list(lam_mul_k)
        self.lam_residual = bool(lam_residual)
        self.lam_concat = bool(lam_concat)
        self.use_scale = bool(use_scale)
        self.upsample_mode = str(unsampling_mode)
        self.scale_factor = int(scale_factor)
        assert att_norm_cfg is None or isinstance(att_norm_cfg, dict)
        self.frozen = bool(frozen)
        self.overflow = 0
        assert 0 <= lam_mul and lam_mul <= 1
        for i in range(len(self.lam_mul_k)):
            self.lam_mul_k[i] = min(self.lam_mul_k[i], 10) if self.lam_mul_k[i] >= 0 else -1

        if self.lam_concat:
            self.qk_channel = self.in_channel+1
        else:
            self.qk_channel = self.in_channel

        self.conv_q = ConvModule(
            in_channels=self.qk_channel,
            out_channels=self.inter_channels,
            kernel_size=1, stride=1, padding=0,
            groups=1, bias='auto',
            norm_cfg=att_norm_cfg,
        )
        self.conv_k = ConvModule(
            in_channels=self.qk_channel,
            out_channels=self.inter_channels,
            kernel_size=1, stride=1, padding=0,
            groups=1, bias='auto',
            norm_cfg=att_norm_cfg,
        )

        self.conv_v = nn.Conv2d(in_channels=self.qk_channel, out_channels=1, kernel_size=1, stride=1)

        self.init_weights()
        if self.frozen:
            self._freeze()

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(AdaptiveMask, self).init_weights()
            return
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
                constant_init(m, val=1, bias=0)

    def _freeze(self):
        if self.frozen:
            # mixblock
            for param in self.conv_q.parameters():
                param.requires_grad = False
            for param in self.conv_k.parameters():
                param.requires_grad = False
            for param in self.conv_v.parameters():
                param.requires_grad = False

    def caculate_attention(self,f_map, f_map_):
        """ caculate attention in samples"""
        b, c, w, h = f_map.size()

        q, k = self.conv_q(f_map).view(b, self.inter_channels, -1).permute(0, 2, 1), \
            self.conv_k(f_map).view(b, self.inter_channels, -1)
        q_, k_ = self.conv_q(f_map_).view(b, self.inter_channels, -1).permute(0, 2, 1), \
            self.conv_k(f_map_).view(b, self.inter_channels, -1)

        # step 2. attn = attn_ = [b,64,64]
        attn = torch.matmul(
            q.type(torch.float32), k_.type(torch.float32)).type(torch.float32)
        attn_ = torch.matmul(
            q_.type(torch.float32), k.type(torch.float32)).type(torch.float32)

        if torch.any(torch.isinf(attn)):
            attn = attn.type(torch.float32).clamp(min=-1e25, max=1e25)
            self.overflow += 1
            # if self.overflow > 10:
            #     raise ValueError("Mistake!")
        if torch.any(torch.isinf(attn_)):
            attn_ = attn_.type(torch.float32).clamp(min=-1e25, max=1e25)
            self.overflow += 1
            # if self.overflow > 10:
            #     raise ValueError("Mistake!")

        if self.use_scale:
            attn /= self.inter_channels ** 0.5
            attn_ /= self.inter_channels ** 0.5

        attn_soft, attn_soft_ = attn.type(torch.float32).softmax(dim=-1), attn_.type(torch.float32).softmax(dim=-1)
        return attn_soft, attn_soft_

    def co_caculate_attention(self, feature_map):
        """ caculate attention in co-samples"""
        b, c, w, h = feature_map[0].size()
        q = []
        k = []
        attn =[]
        for i in range(0, len(feature_map)):
            q.append(self.conv_q(feature_map[i]).view(b, self.inter_channels, -1).permute(0, 2, 1))
            k.append(self.conv_k(feature_map[i]).view(b, self.inter_channels, -1))

        for i in range(0, len(q)):
            b, c, p = k[0].size()
            co_attn = torch.zeros(b, p, p).cuda()
            for j in range(0, len(k)):
                if i != j:
                   co_attn += torch.matmul(q[i].type(torch.float32), k[j].type(torch.float32))
            if torch.any(torch.isinf(co_attn)):
                co_attn = co_attn.type(torch.float32).clamp(min=-1e25, max=1e25)
            if self.use_scale:
                co_attn /= self.inter_channels ** 0.5

            co_attn = co_attn.type(torch.float32).softmax(dim=-1)
            attn.append(co_attn)

        return attn

    def normal_mix(self, f_map, f_map_, lam):

        b, c, w, h = f_map.size()

        results = dict(f_map=f_map, f_map_=f_map_)

        # step 0. generate mask with lam
        if self.lam_concat:
            lam_block = torch.zeros([b, 1, h, w]).to(f_map)
            lam_block[:] = lam
            f_map = torch.cat([f_map, lam_block], dim=1)
            f_map_ = torch.cat([f_map_, 1 - lam_block], dim=1)

        # step 1. define q, k and v ; q=k=[b,c,h*w] v=[b,1,h*w]
        v, v_ = f_map, f_map_
        v = self.conv_v(v).view(b, 1, -1).permute(0, 2, 1)
        v_ = self.conv_v(v_).view(b, 1, -1).permute(0, 2, 1)

        attn, attn_ = self.caculate_attention(f_map, f_map_)

        debug_plot = dict(value=v.view(b, h, -1).clone().detach())
        debug_plot["pairwise_weight"] = attn.clone().detach()
        results["debug_plot"] = debug_plot

        # attn @ v
        mask_lam = torch.matmul(
            attn.type(torch.float32), v.type(torch.float32)
        ).view(b, 1, h, w)
        mask_lam_ = torch.matmul(
            attn_.type(torch.float32), v_.type(torch.float32)
        ).view(b, 1, h, w)

        # step 3. add mask and upsample to [b,2,32,32]
        mask = torch.cat([mask_lam, mask_lam_], dim=1)
        if torch.any(torch.isnan(mask)):
            # print("is nan")
            mask_lam = torch.matmul(attn.type(torch.float64), v.type(torch.float64)).reshape(b, 1, w, h)
            mask_lam_ = torch.matmul(attn_.type(torch.float64), v_.type(torch.float64)).reshape(b, 1, w, h)
            mask_lam = torch.where(torch.isnan(mask_lam), torch.full_like(mask_lam, 1e-4), mask_lam)
            mask_lam_ = torch.where(torch.isnan(mask_lam_), torch.full_like(mask_lam_, 1e4), mask_lam_)
            mask = torch.cat([mask_lam, mask_lam_], dim=1)
            self.overflow += 1
            # if self.overflow > 10:
            #     raise ValueError("Mistake!")

        return mask, results

    def co_mix(self, feature_map, lam):
        b, c, w, h = feature_map[0].size()

        results = dict()

        # step 0. generate mask with lam
        if self.lam_concat:
            for i in range(0, len(feature_map)):
                lam_block = torch.zeros([b,1,h,w]).to(feature_map[i])
                lam_block[:] = lam[i]
                feature_map[i] = torch.cat([feature_map[i], lam_block], dim=1)

        # step 1. define q, k and v ; q=k=[b,c,h*w] v=[b,1,h*w]
        v = []
        # upsample mode
        for i in range(0, len(feature_map)):
            v.append(feature_map[i])
            v[i] = self.conv_v(v[i]).view(b, 1, -1).permute(0, 2, 1)

        attn = self.co_caculate_attention(feature_map)

        debug_plot = dict(value=v[0].view(b, h, -1).clone().detach())
        debug_plot["pairwise_weight"] = attn[0].clone().detach()
        results["debug_plot"] = debug_plot

        # attn @ v
        mask_zero = torch.matmul(attn[0].type(torch.float32), v[0].type(torch.float32)).view(b, 1, h, w)
        mask_list = [mask_zero, ]
        for i in range(1, len(v)):
            mask_lam = torch.matmul(attn[i].type(torch.float32), v[i].type(torch.float32)).view(b, 1, h, w)
            mask_list.append(mask_lam)
            if i == 1:
                mask = torch.cat([mask_zero, mask_list[i]], dim=1)
            else:
                mask = torch.cat([mask, mask_list[i]], dim=1)

        # step 3. add mask and upsample to [b,2,32,32]
        if torch.any(torch.isnan(mask)):
            print("nan")
            for i in range(0, len(mask_list)):
                mask_list[i] = torch.matmul(attn[i].type(torch.float64), v[i].type(torch.float64)).reshape(b, 1, w, h)
                mask_list[i] = torch.where(torch.isnan(mask_list[i]), torch.full_like(mask_list[i], 0.3), mask_list[i])
                mask_zero = mask_list[0]
                if i == 1:
                    mask = torch.cat([mask_zero, mask_list[i]], dim=1)
                elif i > 1:
                    mask = torch.cat([mask, mask_list[i]], dim=1)
        return mask, results

    def forward(self, f_maps, lam):
        # random choice upsample
        if self.upsample_mode == 'None':
            mode=['bilinear', 'nearest', 'bicubic']
            self.upsample_mode = random.choice(mode)

        results = dict()
        # two samples
        if len(f_maps) == 2:
            f_map, f_map_ = f_maps[0], f_maps[1]
            mask, results = self.normal_mix(f_map, f_map_, lam)
        elif len(f_maps) > 2:
            mask, results = self.co_mix(f_maps, lam)

        mask = F.interpolate(mask, scale_factor=self.scale_factor, mode=self.upsample_mode)
        mask = mask.softmax(dim=1)

        results["mask"] = mask
        return results
