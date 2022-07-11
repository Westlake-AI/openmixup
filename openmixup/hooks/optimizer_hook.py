import re
from mmcv.runner import allreduce_grads, OptimizerHook
from mmcv.runner import Fp16OptimizerHook as _Fp16OptimizerHook
from mmcv.utils import TORCH_VERSION, _BatchNorm, digit_version

try:
    import apex
    has_apex = True
except:
    has_apex = False


class DistOptimizerHook(OptimizerHook):
    """Optimizer hook for distributed training.
    
    Args:
        update_interval (int): Frequency of epochs to call the hook. Default: 1.
        cancel_grad (dict): Config dict for cancelling gradients for selected parameters,
            e.g., cancel_grad=dict(regexp=cancel_iter), 'regexp' stands for param_name.
            Default: None.
        grad_clip (dict, optional): Dict to config the value of grad clip.
            E.g., grad_clip = dict(max_norm=10). Defaults to None.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
        use_fp16 (bool): Whether to use fp16 training skills. Defalut: False.
    """

    def __init__(self,
                 update_interval=1,
                 cancel_grad=None,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 use_fp16=False,
                ):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.use_fp16 = use_fp16
        self.initialized = False

        # basic args
        if use_fp16 and has_apex == False:
            print('Optimizer: apex is not installed! '
                  'Please use Fp16OptimizerHook supported by mmcv=>1.1.4.')
        if cancel_grad is not None:
            self.cancel_grad = dict()
            if isinstance(cancel_grad, dict):
                self.cancel_grad.update(cancel_grad)
        else:
            self.cancel_grad = None

    def has_batch_norm(self, module):
        if isinstance(module, _BatchNorm):
            return True
        for m in module.children():
            if self.has_batch_norm(m):
                return True
        return False

    def _init(self, runner):
        if runner.iter % self.update_interval != 0:
            runner.logger.warning(
                'Resume iter number is not divisible by update_interval in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )
        if self.has_batch_norm(runner.model) and self.update_interval > 1:
            runner.logger.warning(
                'GradientCumulativeOptimizerHook may slightly decrease '
                'performance if the model has BatchNorm layers.')

        residual_iters = runner.max_iters - runner.iter
        self.divisible_iters = (
            residual_iters // self.update_interval * self.update_interval)
        self.remainder_iters = residual_iters - self.divisible_iters

        self.initialized = True

    def before_run(self, runner):
        runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        # In some cases, MMCV's GradientCumulativeOptimizerHook will
        # cause the loss_factor to be zero and we fix this bug in our
        # implementation.

        if not self.initialized:
            self._init(runner)

        if runner.iter < self.divisible_iters:
            loss_factor = self.update_interval
        else:
            loss_factor = self.remainder_iters
        runner.outputs['loss'] /= loss_factor

        if self.use_fp16 and has_apex:
            with apex.amp.scale_loss(
                runner.outputs['loss'], runner.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            runner.outputs['loss'].backward()

        if (self.every_n_iters(runner, self.update_interval)
                or self.is_last_iter(runner)):
            # cancel gradients of selected params
            if self.cancel_grad is not None:
                for regexp, cancel_iter in self.cancel_grad.items():
                    if runner.iter < cancel_iter:
                        for name, p in runner.model.module.named_parameters():
                            if re.search(regexp, name):
                                p.grad = None
            # clip gradients
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            # update
            runner.optimizer.step()
            runner.optimizer.zero_grad()


if (TORCH_VERSION != 'parrots'
        and digit_version(TORCH_VERSION) >= digit_version('1.6.0')):

    class Fp16OptimizerHook(_Fp16OptimizerHook):
        """FP16 optimizer hook (using PyTorch's implementation).

        The steps of fp16 optimizer is as follows.
        1. Scale the loss value.
        2. BP in the fp16 model.
        2. Copy gradients from fp16 model to fp32 weights.
        3. Update fp32 weights.
        4. Copy updated parameters from fp32 weights to fp16 model.

        Refer to https://arxiv.org/abs/1710.03740 for more details.

        Args:
            update_interval (int): Frequency of epochs to call the hook. Default: 1.
            cancel_grad (dict): Config dict for cancelling gradients for selected
                parameters, e.g., cancel_grad=dict(regexp=cancel_iter), 'regexp' stands
                for param_name. Default: None.
            grad_clip (dict): Gradient clip tricks. Default: None.
            loss_scale (float | str | dict): Scale factor multiplied with loss.
                If loss_scale is a float, static loss scaling will be used with
                the specified scale. If loss_scale is a string, it must be
                'dynamic', then dynamic loss scaling will be used.
                It can also be a dict containing arguments of LossScaler.
                Defaults to 512.
        """

        def __init__(self,
                     update_interval=1,
                     cancel_grad=None,
                     **kwargs):
            super(Fp16OptimizerHook, self).__init__(**kwargs)
            self.update_interval = update_interval
            if cancel_grad is not None:
                assert isinstance(cancel_grad, dict)
                self.cancel_grad = cancel_grad
            else:
                self.cancel_grad = None

        def after_train_iter(self, runner):
            """Backward optimization steps for Mixed Precision Training. For
            dynamic loss scaling, please refer `loss_scalar.py`

            1. Scale the loss by a scale factor.
            2. Backward the loss to obtain the gradients (fp16).
            3. Copy gradients from the model to the fp32 weight copy.
            4. Scale the gradients back and update the fp32 weight copy.
            5. Copy back the params from fp32 weight copy to the fp16 model.
            """
            # scale the loss value
            runner.outputs['loss'] /= self.update_interval
            self.loss_scaler.scale(runner.outputs['loss']).backward()

            if self.every_n_iters(runner, self.update_interval):
                # cancel gradients of selected params
                if self.cancel_grad is not None:
                    for regexp, cancel_iter in self.cancel_grad.items():
                        if runner.iter < cancel_iter:
                            for name, p in runner.model.module.named_parameters():
                                if re.search(regexp, name):
                                    p.grad = None

                # copy fp16 grads in the model to fp32 params in the optimizer
                self.loss_scaler.unscale_(runner.optimizer)

                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(runner.model.parameters())
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                                runner.outputs['num_samples'])

                # backward and update scaler
                self.loss_scaler.step(runner.optimizer)
                self.loss_scaler.update(self._scale_update_param)

                # save state_dict of loss_scaler
                runner.meta.setdefault(
                    'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

                # clear grads of last iteration
                runner.model.zero_grad()
                runner.optimizer.zero_grad()

else:

    class Fp16OptimizerHook(_Fp16OptimizerHook):
        """Fp16 optimizer hook (using mmcv's implementation).

        The steps of fp16 optimizer is as follows.
        1. Scale the loss value.
        2. BP in the fp16 model.
        2. Copy gradients from fp16 model to fp32 weights.
        3. Update fp32 weights.
        4. Copy updated parameters from fp32 weights to fp16 model.

        Refer to https://arxiv.org/abs/1710.03740 for more details.

        Args:
            update_interval (int): Frequency of epochs to call the hook. Default: 1.
            cancel_grad (dict): Config dict for cancelling gradients for selected
                parameters, e.g., cancel_grad=dict(regexp=cancel_iter), 'regexp' stands
                for param_name. Default: None.
            grad_clip (dict): Gradient clip tricks. Default: None.
            loss_scale (float | str | dict): Scale factor multiplied with loss.
                If loss_scale is a float, static loss scaling will be used with
                the specified scale. If loss_scale is a string, it must be
                'dynamic', then dynamic loss scaling will be used.
                It can also be a dict containing arguments of LossScaler.
                Defaults to 512.
        """

        def __init__(self,
                     update_interval=1,
                     cancel_grad=None,
                     **kwargs):
            super(Fp16OptimizerHook, self).__init__(**kwargs)
            self.update_interval = update_interval
            if cancel_grad is not None:
                assert isinstance(cancel_grad, dict)
                self.cancel_grad = cancel_grad
            else:
                self.cancel_grad = None

        def after_train_iter(self, runner):
            """Backward optimization steps for Mixed Precision Training. For
            dynamic loss scaling, please refer `loss_scalar.py`

            1. Scale the loss by a scale factor.
            2. Backward the loss to obtain the gradients (fp16).
            3. Copy gradients from the model to the fp32 weight copy.
            4. Scale the gradients back and update the fp32 weight copy.
            5. Copy back the params from fp32 weight copy to the fp16 model.
            """
            # scale the loss value
            runner.outputs['loss'] /= self.update_interval
            scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
            scaled_loss.backward()

            if self.every_n_iters(runner, self.update_interval):
                # cancel gradients of selected params
                if self.cancel_grad is not None:
                    for regexp, cancel_iter in self.cancel_grad.items():
                        if runner.iter < cancel_iter:
                            for name, p in runner.model.module.named_parameters():
                                if re.search(regexp, name):
                                    p.grad = None

                # copy fp16 grads in the model to fp32 params in the optimizer
                fp32_weights = []
                for param_group in runner.optimizer.param_groups:
                    fp32_weights += param_group['params']
                self.copy_grads_to_fp32(runner.model, fp32_weights)
                # allreduce grads
                if self.distributed:
                    allreduce_grads(fp32_weights, self.coalesce,
                                    self.bucket_size_mb)

                has_overflow = self.loss_scaler.has_overflow(fp32_weights)
                # if has overflow, skip this iteration
                if not has_overflow:
                    # scale the gradients back
                    for param in fp32_weights:
                        if param.grad is not None:
                            param.grad.div_(self.loss_scaler.loss_scale)
                    if self.grad_clip is not None:
                        grad_norm = self.clip_grads(fp32_weights)
                        if grad_norm is not None:
                            # Add grad norm to the logger
                            runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                                    runner.outputs['num_samples'])
                    # update fp32 params
                    runner.optimizer.step()
                    # copy fp32 params to the fp16 model
                    self.copy_params_to_fp16(runner.model, fp32_weights)
                else:
                    runner.logger.warning(
                        'Check overflow, downscale loss scale '
                        f'to {self.loss_scaler.cur_scale}')

                self.loss_scaler.update_scale(has_overflow)

                # save state_dict of loss_scaler
                runner.meta.setdefault(
                    'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

                # clear grads of last iteration
                runner.model.zero_grad()
                runner.optimizer.zero_grad()
