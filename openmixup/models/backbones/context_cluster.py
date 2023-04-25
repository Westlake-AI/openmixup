import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init

from ..registry import BACKBONES
from ..utils import to_2tuple
from .base_backbone import BaseBackbone


class Mlp(nn.Module):
    """An implementation of vanilla FFN

    Args:
        in_features (int): The feature dimension.
        hidden_features (int): The hidden dimension of FFNs.
        out_features (int): The output dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        x = self.drop(x)
        return x


class PointRecuder(nn.Module):
    """ Point Reducer is implemented by a layer of conv since it is mathmatically equal.
        Input: tensor in shape [B, in_chans, H, W]
        Output: tensor in shape [B, embed_dim, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """ Pair-wise similarity matrix between two tensors
        :param x1: [B,...,M,D]
        :param x2: [B,...,N,D]
        :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Cluster(nn.Module):

    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2,
                 heads=4, head_dim=24, return_center=False):
        """
        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(
            heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center

    def forward(self, x):
        value = self.v(x)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)

        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            _, _, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map ({w0}x{h0}) can be divided by the fold {self.fold_w}x{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h",
                          f1=self.fold_w, f2=self.fold_h)  # [bxblocks, c, ks[0], ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h",
                              f1=self.fold_w, f2=self.fold_h)

        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # [b, c, C_W, C_H], setting M = C_W x C_H and N = w x h
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
        b, c, cw, ch = centers.shape
        sim = torch.sigmoid(  # [B, M, N]
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )
        # we use mask to assign each point to one center sololy
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary mask [B, M, N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B, N, D]

        # aggregate step, out shape [B,M,D]
        # Update Comment: Mar/26/2022
        #  a small bug: mask.sum should be sim.sum according to Eq. (1), mask can be considered as a hard
        #  version of sim in out implementation. We will update all checkpoints and all models are re-trained.
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers
               ) / (mask.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=cw)
        else:
            # dispatch step, return to each point in a cluster
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B, N, D]
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)",
                            f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)

        return out


class ClusterBlock(nn.Module):
    """Implementation of Clustering block"""

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='GN', num_groups=1),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2,
                 heads=4, head_dim=24, return_center=False):
        super().__init__()
        self.embed_dims = dim

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.token_mixer = Cluster(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                   fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim,
                                   return_center=return_center)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_cfg=act_cfg, ffn_drop=drop_rate)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale_1, self.layer_scale_2 = None, None

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (
                nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                constant_init(m, val=1, bias=0)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, depths, mlp_ratio=4.,
                 act_cfg=dict(type='GELU'), norm_cfg=dict(type='GN', num_groups=1),
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2,
                 heads=4, head_dim=24, return_center=False):
    blocks = []
    for block_idx in range(depths[index]):
        block_dpr = drop_path_rate * ( block_idx + sum(depths[:index])) / (sum(depths) - 1)
        blocks.append(
            ClusterBlock(
                dim, mlp_ratio=mlp_ratio,
                act_cfg=act_cfg, norm_cfg=norm_cfg,
                drop_rate=drop_rate, drop_path_rate=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                heads=heads, head_dim=head_dim, return_center=return_center
        ))
    blocks = nn.Sequential(*blocks)

    return blocks


@BACKBONES.register_module()
class ContextCluster(BaseBackbone):
    """ ContextCluster

    A PyTorch implement of : `Image as Set of Points
    <https://openreview.net/forum?id=awnvqZja69>`_

    Modified from the `official repo
    <https://github.com/ma-xu/context-cluster>`_

    Args:
        arch (str | dict): UniFormer architecture.
            If use string, choose from 'small' and 'base'.
            If use dict, it should have below keys:

            - **embed_dims** (List[int]): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **mlp_ratios** (int): The ratio of mlp hidden dim.
            - **patch_strides** (List[int]): The stride of each stage.
            - **conv_stem** (bool): Whether to use conv-stem.

            We provide UniFormer-Tiny (based on VAN-Tiny) in addition to the
            original paper. Defaults to 'small'.

        in_patch_size (int): Specify the patch embedding for the input image.
        down_patch_size (int): Specify the downsample (patch embed).
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
        layer_scale_init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        drop_rate (float): Probability of an element to be dropped. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [32, 64, 196, 320],
                         'depths': [3, 4, 5, 2],
                         'mlp_ratios': [8, 8, 4, 4],
                         'downsamples': [True, True, True, True],
                         'heads': [4, 4, 8, 8],
                         'head_dim': [24, 24, 24, 24],
                        }),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [2, 2, 6, 2],
                         'mlp_ratios': [8, 8, 4, 4],
                         'downsamples': [True, True, True, True],
                         'heads': [4, 4, 8, 8],
                         'head_dim': [32, 32, 32, 32],
                        }),
        **dict.fromkeys(['m', 'medium'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [4, 4, 12, 4],
                         'mlp_ratios': [8, 8, 4, 4],
                         'downsamples': [True, True, True, True],
                         'heads': [6, 6, 12, 12],
                         'head_dim': [32, 32, 32, 32],
                        }),
        **dict.fromkeys(['b', 'base_dim64'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [6, 6, 24, 6],
                         'mlp_ratios': [8, 8, 4, 4],
                         'downsamples': [True, True, True, True],
                         'heads': [8, 8, 16, 16],
                         'head_dim': [32, 32, 32, 32],
                        }),
        **dict.fromkeys(['b96', 'base_dim96'],
                        {'embed_dims': [96, 192, 384, 768],
                         'depths': [4, 4, 12, 4],
                         'mlp_ratios': [8, 8, 4, 4],
                         'downsamples': [True, True, True, True],
                         'heads': [8, 8, 16, 16],
                         'head_dim': [32, 32, 32, 32],
                        }),
    }  # yapf: disable

    def __init__(self,
                 arch='small',
                 in_channels=3,
                 out_indices=-1,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 in_patch_size=4,
                 in_stride=4,
                 in_pad=0,
                 down_patch_size=3,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg=None,
                 # the parameters for context-cluster
                 proposal_w=[2, 2, 2, 2], proposal_h=[2, 2, 2, 2],
                 fold_w=[8, 4, 2, 1], fold_h=[8, 4, 2, 1],
                 return_center=False,
                 **kwargs):
        super(ContextCluster, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
            self.arch = arch.split("-")[0]
        else:
            essential_keys = {
                'embed_dims', 'depths', 'mlp_ratios', 'downsamples', 'heads', 'head_dim',
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch
            self.arch = 'small'

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.mlp_ratios = self.arch_settings['mlp_ratios']
        self.downsamples = self.arch_settings['downsamples']
        self.heads = self.arch_settings['heads']
        self.head_dim = self.arch_settings['head_dim']
        self.num_stages = len(self.depths)
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.patch_embed = PointRecuder(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=in_channels+2, embed_dim=self.embed_dims[0])

        network = []
        for i, _ in enumerate(self.depths):
            stage = basic_blocks(
                self.embed_dims[i], i, self.depths,
                mlp_ratio=self.mlp_ratios[i],
                act_cfg=act_cfg, norm_cfg=norm_cfg,
                drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                proposal_w=proposal_w[i], proposal_h=proposal_h[i],
                fold_w=fold_w[i], fold_h=fold_h[i],
                heads=self.heads[i], head_dim=self.head_dim[i],
                return_center=return_center)
            network.append(stage)
            if i >= self.num_stages - 1:
                break
            if self.downsamples[i] or self.embed_dims[i] != self.embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PointRecuder(
                        patch_size=down_patch_size, stride=2, padding=max(0, down_patch_size-2),
                        in_chans=self.embed_dims[i], embed_dim=self.embed_dims[i + 1]
                ))
        self.network = nn.ModuleList(network)
        assert isinstance(out_indices, (int, tuple, list))
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = len(self.network) + index
                assert 0 <= out_indices[i] < len(self.network), f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # final norm
        norm_layer = build_norm_layer(norm_cfg, self.embed_dims[-1])[1]
        self.add_module(f'norm', norm_layer)

        self._freeze_stages()

    def init_weights(self, pretrained=None):
        super(ContextCluster, self).init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is not None:
                return
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    trunc_normal_init(m, std=0.02, bias=0)
                elif isinstance(m, (
                    nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                    constant_init(m, val=1, bias=0)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            # freeze blocks
            self.network[i].eval()
            for param in self.network[i].parameters():
                param.requires_grad = False

            if i == len(self.network) - 1:
                norm_layer = getattr(self, f'norm')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

    def forward_embeddings(self, x):
        _, _, w, h = x.shape
        # register positional information buffer as two channels.
        range_w = torch.arange(0, w, step=1) / (w - 1.0)
        range_h = torch.arange(0, h, step=1) / (h - 1.0)
        fea_pos = torch.stack(
            torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos.to(x.device)
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)
        x = self.patch_embed(torch.cat([x, pos], dim=1))

        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)

        outs = []
        for i, block in enumerate(self.network):
            x = block(x)

            if i in self.out_indices:
                if i == len(self.network) - 1:
                    norm_layer = getattr(self, f'norm')
                    x = norm_layer(x)
                outs.append(x)

        return outs

    def train(self, mode=True):
        super(ContextCluster, self).train(mode)
        self._freeze_stages()
