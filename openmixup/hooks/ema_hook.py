# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import Hook

from .registry import HOOKS


@HOOKS.register_module()
class EMAHook(Hook):
    r"""Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointSaverHook!

        .. math::
            Xema\_{t+1} = \text{momentum} \times Xema\_{t} +
                (1 - \text{momentum}) \times X_t

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.9999.
        resume_from (str): The checkpoint path. Defaults to None.
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'. Default to None.
        warmup_iters (int): The number of iterations that warmup lasts, i.e.,
            warmup by iteration. Default to 0.
        warmup_ratio (float): Attr used at the beginning of warmup equals to
            warmup_ratio * momentum.
        update_interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
    """

    def __init__(self,
                 momentum=0.9999,
                 resume_from=None,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.9,
                 update_interval=1,
                 **kwargs):
        assert isinstance(update_interval, int) and update_interval > 0
        assert momentum > 0 and momentum < 1
        self.momentum = momentum
        self.regular_momentum = momentum
        self.checkpoint = resume_from
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up!')
            assert warmup_iters > 0 and 0 < warmup_ratio <= 1.0
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.update_interval = update_interval

    def get_warmup_momentum(self, cur_iters):
        if self.warmup == 'constant':
            warmup_m = self.warmup_ratio * self.momentum
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_m = (1 - k) * self.momentum
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_m = k * self.momentum
        return warmup_m
    
    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.param_ema_buffer = {}
        self.model_parameters = dict(model.named_parameters(recurse=True))
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        if self.every_n_iters(runner, self.update_interval):
            curr_iter = runner.iter
            if self.warmup is None or curr_iter > self.warmup_iters:
                self.regular_momentum = self.momentum
            else:
                self.regular_momentum = self.get_warmup_momentum(curr_iter)
            for name, parameter in self.model_parameters.items():
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(self.regular_momentum).add_(
                    parameter.data, alpha=1. - self.regular_momentum)

    def after_train_epoch(self, runner):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        self._swap_ema_parameters()

    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)
