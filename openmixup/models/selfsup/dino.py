# reference: https://github.com/open-mmlab/mmselfsup/tree/master/mmselfsup/models/algorithms
import torch
import torch.nn as nn

from openmixup.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class DINO(BaseModel):
    """DINO.

    This module is proposed in `DINO: Emerging Properties in Self-Supervised
    Vision Transformers <https://arxiv.org/abs/2104.14294>`_.

    *** Requiring Hook: `momentum_update` is adjusted by `CosineScheduleHook`
        after_train_iter in `momentum_hook.py`.

    Args:
        backbone (dict): Config for backbone.
        neck (dict): Config for neck.
        head (dict): Config for head.
        pretrained (str, optional): Path for pretrained model.
            Defaults to None.
        base_momentum (float, optional): Base momentum for momentum update.
            Defaults to 0.99.
        data_preprocessor (dict, optional): Config for data preprocessor.
            Defaults to None.
        init_cfg (list[dict] | dict, optional): Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.99,
                 init_cfg=None,
                 **kwargs):
        super(DINO, self).__init__(init_cfg, **kwargs)
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.student = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.teacher = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        
        self.backbone = self.student[0]
        self.neck = self.student[1]
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.base_momentum = base_momentum
        self.momentum = base_momentum
        self.switch_ema = False

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super(DINO, self).init_weights()

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.student[0].init_weights(pretrained=pretrained) # backbone
        self.student[1].init_weights() # projection
        for param_ol, param_tgt in zip(self.student.parameters(),
                                       self.teacher.parameters()):
            param_tgt.requires_grad = False
            param_tgt.data.copy_(param_ol.data)
        # init the predictor in the head
        self.head.init_weights()

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the target network by hook."""
        for param_ol, param_tgt in zip(self.student.parameters(),
                                       self.teacher.parameters()):
            if not self.switch_ema:  # original momentum update
                param_tgt.data = param_tgt.data * self.momentum + \
                                param_ol.data * (1. - self.momentum)
            else:  # switch EMA
                param_tgt.data = param_ol.data

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list) and len(img) >= 2
        global_crops = torch.cat(img[:2])
        local_crops = torch.cat(img[2:])

        # teacher forward
        teacher_output = self.teacher(global_crops)[0]

        # student forward global
        student_output_global = self.student(global_crops)[0]

        # student forward local
        student_output_local = self.student(local_crops)[0]

        student_output = torch.cat(
            (student_output_global, student_output_local))

        # compute loss
        losses = self.head(student_output, teacher_output)

        return dict(loss=losses)
