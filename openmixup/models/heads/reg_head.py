import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, kaiming_init, normal_init
from mmcv.runner import BaseModule

from ..utils import regression_error, trunc_normal_init
from ..registry import HEADS
from ..builder import build_loss


@HEADS.register_module
class RegHead(BaseModule):
    r"""Simplest regression head, with only one fc layer.
    
    Args:
        with_avg_pool (bool): Whether to use GAP before this head.
        loss (list or dict): Config or List of configs for the regression loss.
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output result.
        act_cfg (None | str): Whether to use the activate function.
        frozen (bool): Whether to freeze the parameters.
    """

    def __init__(self,
                 with_avg_pool=False,
                 loss=dict(type='RegressionLoss', loss_weight=1.0, mode="mse_loss"),
                 in_channels=2048,
                 out_channels=1,
                 act_cfg=None,
                 frozen=False,
                 init_cfg=None):
        super(RegHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert loss is None or isinstance(loss, (dict, list))
        assert act_cfg is None or isinstance(act_cfg, dict)

        # loss
        if loss is None:
            loss = [dict(type="RegressionLoss", loss_weight=1.0, mode="mse_loss")]
        elif isinstance(loss, dict):
            loss = [loss]
        self.criterion_num = 0
        for i in range(len(loss)):
            assert isinstance(loss[i], dict)
            _criterion = build_loss(loss[i])
            self.add_module(str(i), _criterion)
            self.criterion_num += 1
        # activate
        self.act = None
        if act_cfg is not None:
            self.act = build_activation_layer(act_cfg)
        
        # fc layer
        self.fc = nn.Linear(in_channels, out_channels)
        if frozen:
            self.frozen()

    def frozen(self):
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        if self.init_cfg is not None:
            super(RegHead, self).init_weights()
            return
        assert init_linear in ['normal', 'kaiming', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'kaiming':
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)

    def forward(self, x, **kwargs):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features. Multi-stage inputs are acceptable
                but only the last stage will be used to classify. The shape of every
                item should be ``(num_samples, in_channels)``.

        Returns:
            Tensor | list: The inference results.
        """
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        else:
            x = x.reshape(x.size(0), -1)
        x = self.fc(x).squeeze()
        if self.act is not None:
            x = self.act(x)
        return [x]

    def loss(self, score, labels, **kwargs):
        """ regression loss forward
        
        Args:
            score (list): Score should be [tensor] in (N,).
            labels (tuple or tensor): Labels should be tensor (N,) by default.
        """
        losses = dict()
        assert isinstance(score, (tuple, list)) and len(score) == 1
        score = score[0]

        # computing loss
        labels = labels.type_as(score)
        _criterion = getattr(self, "0")
        losses['loss'] = _criterion(score, labels, **kwargs)
        if self.criterion_num > 1:
            for i in range(1, self.criterion_num):
                _criterion = getattr(self, str(i))
                losses['loss'] += _criterion(score, labels, **kwargs)
        # compute error
        losses['mse'], _ = regression_error(score, labels, average_mode='mean')
        
        return losses
