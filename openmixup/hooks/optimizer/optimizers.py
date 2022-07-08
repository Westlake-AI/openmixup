import math

import torch
from mmcv.runner.optimizer.builder import OPTIMIZERS
from torch.optim.optimizer import Optimizer, required
from torch.optim import *


@OPTIMIZERS.register_module()
class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        dampening (float, optional): dampening for momentum (default: 0)
        eta (float, optional): LARS coefficient
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9,
        >>>                  weight_decay=1e-4, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 eta=0.001,
                 nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov, eta=eta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            eta = group['eta']
            nesterov = group['nesterov']
            lr = group['lr']
            lars_exclude = group.get('lars_exclude', False)

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if lars_exclude:
                    local_lr = 1.
                else:
                    weight_norm = torch.norm(p).item()
                    grad_norm = torch.norm(d_p).item()
                    # Compute local learning rate for this layer
                    local_lr = eta * weight_norm / \
                        (grad_norm + weight_decay * weight_norm)

                actual_lr = local_lr * lr
                d_p = d_p.add(p, alpha=weight_decay).mul(actual_lr)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                                torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(-d_p)

        return loss


@OPTIMIZERS.register_module()
class LAMB(Optimizer):
    r"""Implements a pure pytorch variant of FuseLAMB (NvLamb variant) optimizer
    from apex.optimizers.FusedLAMB
    reference: https://github.com/NVIDIA/DeepLearningExamples/blob/master/
    PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py

    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training
    BERT in 76 minutes`_.

    This optimizer code was adapted from the following (starting with latest)
    * https://github.com/HabanaAI/Model-References/blob/
    2b435114fe8e31f159b1d3063b8280ae37af7423/PyTorch/nlp/bert/pretraining/lamb.py
    * https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/
    LanguageModeling/Transformer-XL/pytorch/lamb.py
    * https://github.com/cybertronai/pytorch-lamb

    Use FusedLamb if you can (GPU). The reason for including this variant of Lamb
    is to have a version that is
    similar in behaviour to APEX FusedLamb if you aren't using NVIDIA GPUs or
    cannot install/use APEX.

    In addition to some cleanup, this Lamb impl has been modified to support
    PyTorch XLA and has been tested on TPU.

    Original copyrights for above sources are below.

    Modifications Copyright 2021 Ross Wightman

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm
            (default: 1.0)
        trust_clip (bool): enable LAMBC trust ratio clipping (default: False)
        always_adapt (boolean, optional): Apply adaptive learning rate to 0.0
            weight decay parameter (default: False)

    .. _Large Batch Optimization for Deep Learning - Training BERT in 76
        minutes:
        https://arxiv.org/abs/1904.00962
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0.01,
                 grad_averaging=True,
                 max_grad_norm=1.0,
                 trust_clip=False,
                 always_adapt=False):
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            max_grad_norm=max_grad_norm,
            trust_clip=trust_clip,
            always_adapt=always_adapt)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]['params'][0].device
        one_tensor = torch.tensor(
            1.0, device=device
        )  # because torch.where doesn't handle scalars correctly
        global_grad_norm = torch.zeros(1, device=device)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'Lamb does not support sparse gradients, consider '
                        'SparseAdam instead.')
                global_grad_norm.add_(grad.pow(2).sum())

        global_grad_norm = torch.sqrt(global_grad_norm)
        # FIXME it'd be nice to remove explicit tensor conversion of scalars
        #  when torch.where promotes
        # scalar types properly https://github.com/pytorch/pytorch/issues/9190
        max_grad_norm = torch.tensor(
            self.defaults['max_grad_norm'], device=device)
        clip_global_grad_norm = torch.where(global_grad_norm > max_grad_norm,
                                            global_grad_norm / max_grad_norm,
                                            one_tensor)

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0
            beta3 = 1 - beta1 if grad_averaging else 1.0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or
            # pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            if bias_correction:
                bias_correction1 = 1 - beta1**group['step']
                bias_correction2 = 1 - beta2**group['step']
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.div_(clip_global_grad_norm)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient valuesa
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=beta3)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2)  # v_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group['eps'])
                update = (exp_avg / bias_correction1).div_(denom)

                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    update.add_(p, alpha=weight_decay)

                if weight_decay != 0 or group['always_adapt']:
                    # Layer-wise LR adaptation. By default, skip adaptation on
                    # parameters that are
                    # excluded from weight decay, unless always_adapt == True,
                    # then always enabled.
                    w_norm = p.norm(2.0)
                    g_norm = update.norm(2.0)
                    # FIXME nested where required since logical and/or not
                    #  working in PT XLA
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, w_norm / g_norm, one_tensor),
                        one_tensor,
                    )
                    if group['trust_clip']:
                        # LAMBC trust clipping, upper bound fixed at one
                        trust_ratio = torch.minimum(trust_ratio, one_tensor)
                    update.mul_(trust_ratio)

                p.add_(update, alpha=-group['lr'])

        return loss
