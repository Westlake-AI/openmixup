import torch.nn as nn

from openmixup.utils import print_log

from .. import builder
from ..registry import MODELS


@MODELS.register_module
class Representation(nn.Module):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 head_keys=["head0"],
                 pretrained=None,
                ):
        super(Representation, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            self.neck = None
        assert head is None
        self.head_keys = head_keys
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model ckpt from {}.'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)

    def forward_backbone(self, img):
        """Forward backbone

        Returns:
            x (tuple): backbone outputs
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        raise NotImplementedError

    def forward_test(self, img, **kwargs):
        x = self.forward_backbone(img)  # tuple
        if self.neck is not None:
            x = self.neck(x)
        if len(self.head_keys) > 1:
            assert len(self.head_keys) == len(x)
        else:
            keys = ['head0']
        # out_tensors = [out.cpu() for out in outs]  # NxC
        out_tensors = [out.cpu() for out in x]  # NxC
        return dict(zip(keys, out_tensors))

    def aug_test(self, imgs):
        raise NotImplementedError

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            # return self.forward_train(img, **kwargs)
            raise Exception("No such mode: {} in Reprensentation.".format(mode))
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            # return self.forward_backbone(img)
            raise Exception("No such mode: {} in Reprensentation.".format(mode))
        else:
            raise Exception("No such mode: {} in Reprensentation.".format(mode))
