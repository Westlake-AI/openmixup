from math import cos, pi

from mmcv.parallel import is_module_wrapper
from mmcv.runner import Hook

from openmixup.utils import print_log
from .registry import HOOKS


@HOOKS.register_module
class CosineHook(Hook):
    """Hook for Momentum update: Cosine.

    This hook includes momentum adjustment with cosine scheduler:
        m = 1 - ( 1- m_0) * (cos(pi * k / K) + 1) / 2,
        k: current step, K: max adjust steps.

    Args:
        end_momentum (float): The final momentum coefficient for the target
            network. Default: 1.
        adjust_scope (float): Ranging from (0, 1], only adjust momentum in
            this scope. Default: 1.0.
        restart_step (int): Set the momentum to 0 when hit the restart_step
            (by interval), i.e., cut_iter Mod restart_step == 0.
            Default: 1e10 (never restart).
    """

    def __init__(self,
                end_momentum=1.,
                adjust_scope=1.,
                restart_step=1e11,
                update_interval=1, **kwargs):
        self.end_momentum = end_momentum
        self.adjust_scope = adjust_scope
        self.update_interval = update_interval
        self.restart_step = int(min(max(restart_step, 1), 1e10))
        self.run_momentum_update = False
        assert adjust_scope >= 0.

    def before_run(self, runner):
        assert hasattr(runner.model.module, 'momentum'), \
            "The runner must have attribute \"momentum\" in Momentum Hook."
        assert hasattr(runner.model.module, 'base_momentum'), \
            "The runner must have attribute \"base_momentum\" in Momentum Hook."
        if is_module_wrapper(runner.model):
            self.run_momentum_update = hasattr(runner.model.module, 'momentum_update')
        else:
            self.run_momentum_update = hasattr(runner.model, 'momentum_update')
        if self.run_momentum_update:
            print_log("Execute `momentum_update()` after training iter.", logger='root')
        else:
            print_log("Only update `momentum` without `momentum_update()`", logger='root')

    def before_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            if self.adjust_scope < 1:
                max_iter = int(runner.max_iters * self.adjust_scope)
            else:
                max_iter = runner.max_iters
            if cur_iter <= max_iter:
                if cur_iter % self.restart_step == 0:
                    m = 0
                else:
                    base_m = runner.model.module.base_momentum
                    m = self.end_momentum - (self.end_momentum - base_m) * (
                        cos(pi * cur_iter / float(max_iter)) + 1) / 2
                runner.model.module.momentum = m

    def after_train_iter(self, runner):
        if self.run_momentum_update == False:
            return
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                runner.model.module.momentum_update()
            else:
                runner.model.momentum_update()


@HOOKS.register_module
class StepHook(Hook):
    """Hook for Momentum update: Step.

    This hook includes momentum adjustment with step scheduler.

    Args:
        step (list): The list of mile-store for the target network.
            Default: [0.6, 0.9].
        gamma (float): The step size. Default: 0.1.
        adjust_scope (float): range from (0, 1], only adjust momentum in
            this scope. Default: 1.0.
        restart_step (int): Set the momentum to 0 when hit the restart_step
            (by interval), i.e., cut_iter Mod restart_step == 0.
            Default: 1e10 (never restart).
    """

    def __init__(self,
                step=[0.6, 0.9],
                gamma=0.1,
                adjust_scope=1.,
                restart_step=1e11,
                update_interval=1, **kwargs):
        self.step = step
        self.gamma = gamma
        self.adjust_scope = adjust_scope
        self.restart_step = int(min(max(restart_step, 1), 1e10))
        self.update_interval = update_interval
        self.run_momentum_update = False
        assert 0 <= adjust_scope and 0 < gamma < 1

    def before_run(self, runner):
        assert hasattr(runner.model.module, 'momentum'), \
            "The runner must have attribute \"momentum\" in Momentum Hook."
        assert hasattr(runner.model.module, 'base_momentum'), \
            "The runner must have attribute \"base_momentum\" in Momentum Hook."
        if is_module_wrapper(runner.model):
            self.run_momentum_update = hasattr(runner.model.module, 'momentum_update')
        else:
            self.run_momentum_update = hasattr(runner.model, 'momentum_update')
        if self.run_momentum_update:
            print_log("Execute `momentum_update()` after training iter.", logger='root')
        else:
            print_log("Only update `momentum` without `momentum_update()`", logger='root')

    def before_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            if self.adjust_scope < 1:
                max_iter = int(runner.max_iters * self.adjust_scope)
            else:
                max_iter = runner.max_iters
            if cur_iter <= max_iter:
                if cur_iter % self.restart_step == 0:
                    runner.model.module.momentum = 0
                else:
                    base_m = runner.model.module.base_momentum
                    for i in range(len(self.step)):
                        if int(self.step[i] * max_iter) >= cur_iter:
                            m = base_m * (1. - pow(self.gamma, i+1))
                            runner.model.module.momentum = m
                            break
            else:
                pass

    def after_train_iter(self, runner):
        if self.run_momentum_update == False:
            return
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                runner.model.module.momentum_update()
            else:
                runner.model.momentum_update()


@HOOKS.register_module
class CosineScheduleHook(Hook):
    """Hook for Momentum update: Cosine.

    This hook includes momentum adjustment with cosine scheduler:
        m = 1 - ( 1- m_0) * (cos(pi * k / K) + 1) / 2,
        k: current step, K: max adjust steps.

    Args:
        end_momentum (float): The final momentum coefficient for the target
            network. Default: 1.
        adjust_scope (float): Ranging from (0, 1], only adjust momentum in
            this scope. Default: 1.0.
        warming_up (string): Warming up from end_momentum to base_momentum.
            Default: "linear".
        restart_step (int): Set the momentum to 0 when hit the restart_step
            (by interval), i.e., cut_iter Mod restart_step == 0.
            Default: 1e10 (never restart).
    """

    def __init__(self,
                end_momentum=1.,
                adjust_scope=[0, 1],
                warming_up="linear",
                restart_step=1e11,
                update_interval=1, **kwargs):
        self.end_momentum = end_momentum
        self.adjust_scope = adjust_scope
        self.warming_up = warming_up
        self.restart_step = int(min(max(restart_step, 1), 1e10))
        self.update_interval = update_interval
        self.run_momentum_update = False
        assert len(adjust_scope) == 2 and adjust_scope[0] <= adjust_scope[1]

    def before_run(self, runner):
        assert hasattr(runner.model.module, 'momentum'), \
            "The runner must have attribute \"momentum\" in Momentum Hook."
        assert hasattr(runner.model.module, 'base_momentum'), \
            "The runner must have attribute \"base_momentum\" in Momentum Hook."
        if is_module_wrapper(runner.model):
            self.run_momentum_update = hasattr(runner.model.module, 'momentum_update')
        else:
            self.run_momentum_update = hasattr(runner.model, 'momentum_update')
        if self.run_momentum_update:
            print_log("Execute `momentum_update()` after training iter.", logger='root')
        else:
            print_log("Only update `momentum` without `momentum_update()`", logger='root')

    def before_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            base_m = runner.model.module.base_momentum
            assert base_m <= self.end_momentum
            if self.adjust_scope[1] < 1:
                max_iter = int(runner.max_iters * self.adjust_scope[1])
            else:
                max_iter = runner.max_iters
            if self.adjust_scope[0] > 0:
                min_iter = int(runner.max_iters * self.adjust_scope[0])
            else:
                min_iter = 0
            
            if min_iter <= cur_iter and cur_iter <= max_iter:
                if cur_iter % self.restart_step == 0:
                    m = 0
                else:
                    m = self.end_momentum - (self.end_momentum - base_m) * (
                        cos(pi * cur_iter / float(max_iter)) + 1) / 2
                runner.model.module.momentum = m
            else:
                if cur_iter < min_iter:  # end_m to base_m
                    if self.warming_up == "linear":
                        m = self.end_momentum - (self.end_momentum - base_m) * (
                            (min_iter - cur_iter) / min_iter)
                        runner.model.module.momentum = m
                    elif self.warming_up == "constant":
                        runner.model.module.momentum = base_m
                    else:
                        assert self.warming_up in ["linear", "constant"]
                else:
                    pass

    def after_train_iter(self, runner):
        if self.run_momentum_update == False:
            return
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                runner.model.module.momentum_update()
            else:
                runner.model.momentum_update()


@HOOKS.register_module
class StepScheduleHook(Hook):
    """Hook for Momentum update: Step.

    This hook includes momentum adjustment with step scheduler.

    Args:
        end_momentum (float): The final momentum coefficient for the
            target network. Default: 1.
        step (list): The list of mile-store for the target network.
            Default: [0.6, 0.9].
        gamma (float): The step size. Default: 0.1.
        adjust_scope (float): range from (0, 1], only adjust momentum in
            this scope. Default: 1.0.
        warming_up (string): Warming up from end_momentum to base_momentum.
            Default: "linear".
        restart_step (int): Set the momentum to 0 when hit the restart_step
            (by interval), i.e., cut_iter Mod restart_step == 0.
            Default: 1e10 (never restart).
    """

    def __init__(self,
                end_momentum=1.,
                step=[0.6, 0.9],
                gamma=0.1,
                adjust_scope=[0, 1],
                warming_up="linear",
                restart_step=1e11,
                update_interval=1, **kwargs):
        self.end_momentum = end_momentum
        self.step = step
        self.gamma = gamma
        self.adjust_scope = adjust_scope
        self.warming_up = warming_up
        self.restart_step = int(min(max(restart_step, 1), 1e10))
        self.update_interval = update_interval
        self.run_momentum_update = False
        assert 0 <= adjust_scope and 0 < gamma < 1

    def before_run(self, runner):
        assert hasattr(runner.model.module, 'momentum'), \
            "The runner must have attribute \"momentum\" in Momentum Hook."
        assert hasattr(runner.model.module, 'base_momentum'), \
            "The runner must have attribute \"base_momentum\" in Momentum Hook."
        if is_module_wrapper(runner.model):
            self.run_momentum_update = hasattr(runner.model.module, 'momentum_update')
        else:
            self.run_momentum_update = hasattr(runner.model, 'momentum_update')
        if self.run_momentum_update:
            print_log("Execute `momentum_update()` after training iter.", logger='root')
        else:
            print_log("Only update `momentum` without `momentum_update()`", logger='root')

    def before_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            base_m = runner.model.module.base_momentum
            assert base_m < self.end_momentum
            if self.adjust_scope[1] < 1:
                max_iter = int(runner.max_iters * self.adjust_scope[1])
            else:
                max_iter = runner.max_iters
            if self.adjust_scope[0] > 0:
                min_iter = int(runner.max_iters * self.adjust_scope[0])
            else:
                min_iter = 0

            if min_iter <= cur_iter and cur_iter <= max_iter:
                if cur_iter % self.restart_step == 0:
                    runner.model.module.momentum = 0
                else:
                    base_m = runner.model.module.base_momentum
                    for i in range(len(self.step)):
                        if int(self.step[i] * max_iter) >= cur_iter:
                            m = base_m * (self.end_momentum - pow(self.gamma, i+1))
                            runner.model.module.momentum = m
                            break
            else:
                if cur_iter < min_iter:  # end_m to base_m
                    if self.warming_up == "linear":
                        m = self.end_momentum - (self.end_momentum - base_m) * (
                            (min_iter - cur_iter) / min_iter)
                        runner.model.module.momentum = m
                    elif self.warming_up == "constant":
                        runner.model.module.momentum = base_m
                    else:
                        assert self.warming_up in ["linear", "constant"]
                else:
                    pass

    def after_train_iter(self, runner):
        if self.run_momentum_update == False:
            return
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                runner.model.module.momentum_update()
            else:
                runner.model.momentum_update()
