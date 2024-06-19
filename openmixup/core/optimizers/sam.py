import torch
from mmcv.runner.optimizer.builder import OPTIMIZERS
from torch.optim.optimizer import Optimizer, required


class SAM(Optimizer):
    r"""Sharpness-Aware Minimization (SAM) optimizer.

    Implementation of `Sharpness-Aware Minimization for Efficiently Improving Generalization
        (ICLR'2021) <https://openreview.net/forum?id=6Tm1mposlrM>`_.

        https://github.com/davda54/sam
        https://github.com/google-research/sam
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        # assert closure is not None, \
        #     "Sharpness Aware Minimization requires closure, but it was not provided"
        if closure is not None:
            closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
            self.first_step(zero_grad=True)
            closure()
            self.second_step(zero_grad=True)
        else:
            self.second_step(zero_grad=True)

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0
                          ) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


@OPTIMIZERS.register_module()
class SAMAdam(SAM):
    def __init__(self, params, lr=required,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False,
                 rho=0.05, adaptive=False, **kwargs):
        defaults_opt = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(
            params=params, base_optimizer=torch.optim.Adam, rho=rho, adaptive=adaptive, **defaults_opt)


@OPTIMIZERS.register_module()
class SAMAdamW(SAM):
    def __init__(self, params, lr=required,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False,
                 rho=0.05, adaptive=False, **kwargs):
        defaults_opt = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(
            params=params, base_optimizer=torch.optim.AdamW, rho=rho, adaptive=adaptive, **defaults_opt)


@OPTIMIZERS.register_module()
class SAMSGD(SAM):
    def __init__(self, params, lr=required,
                 momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 rho=0.05, adaptive=False):
        defaults_opt = dict(lr=lr, momentum=momentum, dampening=dampening,
                            weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(
            params=params, base_optimizer=torch.optim.SGD, rho=rho, adaptive=adaptive, **defaults_opt)
