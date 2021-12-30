import re
from mmcv.runner import OptimizerHook
try:
    import apex
    has_apex = True
except:
    has_apex = False
    print('Optimizer: apex is not installed')


class DistOptimizerHook(OptimizerHook):
    """Optimizer hook for distributed training.
    
    Args:
        update_interval (int): Frequency of epochs to call the hook. Default: 1.
        cancel_grad (dict): Config dict for cancelling gradients for selected parameters,
            e.g., cancel_grad=dict(regexp=cancel_iter), 'regexp' stands for param_name.
            Default: None.
        grad_clip (dict): Gradient clip tricks. Default: None.
        use_fp16 (bool): Whether to use fp16 training skills. Defalut: False.
    """

    def __init__(self,
                 update_interval=1,
                 cancel_grad=None,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 use_fp16=False):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.use_fp16 = use_fp16
        # basic args
        if use_fp16:
            assert has_apex
        if cancel_grad is not None:
            assert isinstance(cancel_grad, dict)
            self.cancel_grad = cancel_grad
        else:
            self.cancel_grad = None

    def before_run(self, runner):
        runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        runner.outputs['loss'] /= self.update_interval
        if self.use_fp16:
            with apex.amp.scale_loss(runner.outputs['loss'], runner.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            runner.outputs['loss'].backward()
        if self.every_n_iters(runner, self.update_interval):
            # clip gradients
            if self.grad_clip is not None:
                self.clip_grads(runner.model.parameters())
            # cancel gradients of selected params
            if self.cancel_grad is not None:
                cur_iter = runner.iter
                cur_dict = dict()
                for name, p in runner.model.named_parameters():
                    for regexp, cancel_iter in self.cancel_grad.items():
                        if cancel_iter > cur_iter:
                            if re.search(regexp, name):
                                p.grad = None
                                cur_dict[regexp] = cancel_iter
                self.cancel_grad = cur_dict
                if not self.cancel_grad:
                    self.cancel_grad = None
            # update
            runner.optimizer.step()
            runner.optimizer.zero_grad()
