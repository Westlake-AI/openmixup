import torch
import torch.nn as nn

from openmixup.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class SelfTuning(BaseModel):
    """
    Implementation of "Self-Tuning for Data-Efficient Deep Learning
        (https://arxiv.org/pdf/2102.12903.pdf)".
    
    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head_cls: Config dict for classification head.
        queue_size (int): Number of class-specific keys maintained in the queue
            (for the momentum queue). Default: 32.
        proj_dim (int): Dimension of the projector neck (for the momentum queue).
            Default: 128.
        class_num: Total class number of the dataset.
        pretrained: loading from pre-trained model or not (default: True)
        momentum (float): Momentum coefficient for the momentum-updated encoder.
            Default: 0.999.
        temperature (float): The temperature hyper-parameter that controls the
            concentration level of the distribution. Default: 0.07.
    """
    
    def __init__(self,
                 backbone,
                 neck=None,
                 head_cls=None,
                 queue_size=32,
                 proj_dim=128,
                 class_num=200,
                 momentum=0.999,
                 temperature=0.07,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(SelfTuning, self).__init__(init_cfg, **kwargs)
        self.encoder_q = builder.build_backbone(backbone)
        self.encoder_k = builder.build_backbone(backbone)
        self.projector_q = builder.build_neck(neck)
        self.projector_k = builder.build_neck(neck)
        self.backbone = self.encoder_q
        self.head = builder.build_head(head_cls)
        self.init_weights(pretrained=pretrained)

        self.queue_size = queue_size
        self.momentum = momentum
        self.class_num = class_num
        self.pretrained = pretrained
        self.temperature = temperature
        self.KL = nn.KLDivLoss(reduction='batchmean')

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder_q.init_weights(pretrained=pretrained)
        self.projector_q.init_weights(init_linear='kaiming')
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(self.projector_q.parameters(),
                                    self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)

        # create the momentum queue
        self.register_buffer("queue_list", torch.randn(
            proj_dim, queue_size * self.class_num))
        self.queue_list = nn.functional.normalize(self.queue_list, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(
            self.class_num, dtype=torch.long)) # pointer
    
    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        # init q
        if pretrained is not None:
            print_log('load encoder_q from: {}'.format(pretrained), logger='root')
        self.encoder_q.init_weights(pretrained=pretrained)
        self.projector_q.init_weights(init_linear='kaiming')
        self.head.init_weights(init_linear='normal')
        # init k
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projector_q.parameters(),
                                    self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.encoder_q(img)
        return x

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

        for param_q, param_k in zip(self.projector_q.parameters(),
                                    self.projector_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key_c, c):
        """ Update queue of the class """
        # gather keys before updating queue
        batch_size = key_c.shape[0]
        ptr = int(self.queue_ptr[c])
        real_ptr = ptr + c * self.queue_size
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_list[:, real_ptr:real_ptr + batch_size] = key_c.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[c] = ptr

    def forward_pgc(self, im_q, im_k, labels):
        """ forward of PGC loss in Self-Tuning """
        batch_size = im_q.size(0)
        # compute query features
        q_f = self.encoder_q(im_q)[-1] 
        q_c = self.projector_q([q_f])[0] # queries: q_c (N x projector_dim)
        q_c = nn.functional.normalize(q_c, dim=1)

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k_f = self.encoder_k(im_k)[-1] 
            k_c = self.projector_k([k_f])[0] # keys: k_c (N x projector_dim)
            k_c = nn.functional.normalize(k_c, dim=1)

        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nl,nl->n', [q_c, k_c]).unsqueeze(-1)  # Einstein sum is more intuitive

        # cur_queue_list: queue_size * class_num
        cur_queue_list = self.queue_list.clone().detach()

        l_neg_list = torch.Tensor([]).cuda()
        l_pos_list = torch.Tensor([]).cuda()

        for i in range(batch_size):
            neg_sample = torch.cat([cur_queue_list[:, 0:labels[i]*self.queue_size],
                                    cur_queue_list[:, (labels[i]+1)*self.queue_size:]],
                                   dim=1)
            pos_sample = cur_queue_list[:, labels[i]*self.queue_size: (labels[i]+1)*self.queue_size]
            ith_neg = torch.einsum('nl,lk->nk', [q_c[i: i+1], neg_sample])
            ith_pos = torch.einsum('nl,lk->nk', [q_c[i: i+1], pos_sample])
            l_neg_list = torch.cat((l_neg_list, ith_neg), dim=0)
            l_pos_list = torch.cat((l_pos_list, ith_pos), dim=0)
            self._dequeue_and_enqueue(k_c[i: i+1], labels[i])
        
        # logits: 1 + queue_size + queue_size * (class_num - 1)
        PGC_logits = torch.cat([l_pos, l_pos_list, l_neg_list], dim=1)
        # apply temperature
        PGC_logits = nn.LogSoftmax(dim=1)(PGC_logits / self.temperature)

        PGC_labels = torch.zeros([batch_size, 1 + self.queue_size*self.class_num]).cuda()
        PGC_labels[:, 0:self.queue_size+1].fill_(1.0 / (self.queue_size + 1))
        return PGC_logits, PGC_labels, q_f
    
    def forward_train(self, img, gt_labels, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, 4, C, H, W). The first two are
                labeled while the latter two are unlabeled, which are mean centered
                and std scaled.
            gt_labels (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5 and img.size(1) == 4, \
            "Input both must have 5 dims, got: {} and {}".format(img.dim())
        # for labeled data
        img_labeled_q = img[:, 0, ...].contiguous()
        img_labeled_k = img[:, 1, ...].contiguous()
        PGC_logit_labeled, PGC_label_labeled, _ = \
            self.forward_pgc(img_labeled_q, img_labeled_k, gt_labels)
        PGC_loss_labeled = self.KL(PGC_logit_labeled, PGC_label_labeled)

        img_labeled_q = img[:, 0, ...].contiguous()
        x = self.encoder_q(img_labeled_q)[-1]
        outs = self.head([x])
        CE_loss = self.head.loss(outs, gt_labels)

        # for unlabeled data
        img_unlabeled_q = img[:, 2, ...].contiguous()
        img_unlabeled_k = img[:, 3, ...].contiguous()
        with torch.no_grad():  # no gradient for q
            q_f_unlabeled = self.encoder_q(img_unlabeled_q)[-1]
            logit_unlabeled = self.head([q_f_unlabeled])[0]
            prob_unlabeled = torch.softmax(logit_unlabeled.detach(), dim=-1)
            _, predict_unlabeled = torch.max(prob_unlabeled, dim=-1)
        PGC_logit_unlabeled, PGC_label_unlabeled, _ = \
            self.forward_pgc(img_unlabeled_q, img_unlabeled_k, predict_unlabeled)
        PGC_loss_unlabeled = self.KL(PGC_logit_unlabeled, PGC_label_unlabeled)

        # losses
        losses = {
            'loss': CE_loss['loss'] + PGC_loss_labeled + PGC_loss_unlabeled,
            'acc_ce': CE_loss['acc']
        }
        return losses

    def forward_test(self, img, **kwargs):
        """ original classification test """
        x = self.forward_backbone(img)  # tuple
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward_inference(self, img, **kwargs):
        """ inference prediction """
        x = self.encoder_q(img)
        preds_one_k = self.head(x, post_process=True)
        return preds_one_k[0]
