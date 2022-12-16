from mmcv.runner import Hook
from math import cos, pi
from .registry import HOOKS

from openmixup.utils import print_log


class LrAddtionalSchedulerHook(Hook):
    """LR Addtional Scheduler.

    Args:
        addtional_indice (list): A list of indice for selected params.
        by_epoch (bool): Attr changes epoch by epoch. If by_epoch is True, the
            attribute will be updated each epoch when warm_up is none. You can
            set by_epoch to false (i.e., by iter) for more frequently update.
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(self,
                addtional_indice=None,
                by_epoch=True,
                warmup=None,
                warmup_iters=0,
                warmup_ratio=0.1,
                warmup_by_epoch=False,
                update_interval=1,
                **kwargs):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 <= warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range [0,1]'
        
        # optional indice
        self.addtional_indice = addtional_indice
        assert addtional_indice is not None
        # basic lr scheduler args
        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch
        self.update_interval = update_interval

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for optinal param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                j = 0
                for i, param_group in enumerate(optim.param_groups):
                    if i in self.addtional_indice:
                        lr = lr_groups[k][j]
                        param_group['lr'] = lr
                        j += 1
        else:
            j = 0
            for i, param_group in enumerate(runner.optimizer.param_groups):
                if i in self.addtional_indice:
                    lr = lr_groups[j]
                    param_group['lr'] = lr
                    j += 1

    def get_lr(self, runner, base_lr):
        raise NotImplementedError

    def get_regular_lr(self, runner):
        if isinstance(runner.optimizer, dict):
            lr_groups = {}
            for k in runner.optimizer.keys():
                _lr_group = [
                    self.get_lr(runner, _base_lr)
                    for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})
            return lr_groups
        else:
            return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                _base_lr = list()
                for i, group in enumerate(optim.param_groups):
                    if i in self.addtional_indice:
                        group.setdefault('initial_lr', group['lr'])
                        _base_lr.append(group['initial_lr'], group['lr'])
                self.base_lr.update({k: _base_lr})
        else:
            self.base_lr = list()
            for i, group in enumerate(runner.optimizer.param_groups):
                if i in self.addtional_indice:
                    group.setdefault('initial_lr', group['lr'])
                    self.base_lr.append(group['initial_lr'])

    def before_train_epoch(self, runner):
        if self.warmup_by_epoch:
            epoch_len = len(runner.data_loader)
            self.warmup_iters = self.warmup_epochs * epoch_len
        if not self.by_epoch:
            return
        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)


@HOOKS.register_module()
class FixedLrAdditionalHook(LrAddtionalSchedulerHook):

    def __init__(self, **kwargs):
        super(FixedLrAdditionalHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        return base_lr


@HOOKS.register_module()
class StepLrAdditionalHook(LrAddtionalSchedulerHook):

    def __init__(self, step, gamma=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        super(StepLrAdditionalHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        if isinstance(self.step, int):
            return base_lr * (self.gamma**(progress // self.step))

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        return base_lr * self.gamma**exp


@HOOKS.register_module()
class ExpLrAdditionalHook(LrAddtionalSchedulerHook):

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(ExpLrAdditionalHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * self.gamma**progress


@HOOKS.register_module()
class PolyLrAdditionalHook(LrAddtionalSchedulerHook):

    def __init__(self, power=1., min_lr=0., **kwargs):
        self.power = power
        self.min_lr = min_lr
        super(PolyLrAdditionalHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr


@HOOKS.register_module()
class InvLrAdditionalHook(LrAddtionalSchedulerHook):

    def __init__(self, gamma, power=1., **kwargs):
        self.gamma = gamma
        self.power = power
        super(InvLrAdditionalHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_lr * (1 + self.gamma * progress)**(-self.power)


@HOOKS.register_module()
class CosineAnnealingLrAdditionalHook(LrAddtionalSchedulerHook):

    def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super(CosineAnnealingLrAdditionalHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr
        return annealing_cos(base_lr, target_lr, progress / max_progress)


@HOOKS.register_module()
class CosineRestartLrAdditionalHook(LrAddtionalSchedulerHook):
    """Cosine annealing with restarts learning rate scheme.

    Args:
        periods (list[int]): Periods for each cosine anneling cycle.
        restart_weights (list[float], optional): Restart weights at each
            restart iteration. Default: [1].
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self,
                 periods,
                 restart_weights=[1],
                 min_lr=None,
                 min_lr_ratio=None,
                 **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        super(CosineRestartLrAdditionalHook, self).__init__(**kwargs)

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
        else:
            progress = runner.iter

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        idx = get_position_from_periods(progress, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((progress - nearest_restart) / current_periods, 1)
        return annealing_cos(base_lr, target_lr, alpha, current_weight)


def get_position_from_periods(iteration, cumulative_periods):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_periods = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_periods (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_periods):
        if iteration <= period:
            return i
    raise ValueError(f'Current iteration {iteration} exceeds '
                     f'cumulative_periods {cumulative_periods}')


@HOOKS.register_module()
class CyclicLrAdditionalHook(LrAddtionalSchedulerHook):
    """Cyclic LR Scheduler.

    Implement the cyclical learning rate policy (CLR) described in
    https://arxiv.org/pdf/1506.01186.pdf

    Different from the original paper, we use cosine anealing rather than
    triangular policy inside a cycle. This improves the performance in the
    3D detection area.

    Attributes:
        target_ratio (tuple[float]): Relative ratio of the highest LR and the
            lowest LR to the initial LR.
        cyclic_times (int): Number of cycles during training
        step_ratio_up (float): The ratio of the increasing process of LR in
            the total cycle.
        by_epoch (bool): Whether to update LR by epoch.
    """

    def __init__(self,
                 by_epoch=False,
                 target_ratio=(10, 1e-4),
                 cyclic_times=1,
                 step_ratio_up=0.4,
                 **kwargs):
        if isinstance(target_ratio, float):
            target_ratio = (target_ratio, target_ratio / 1e5)
        elif isinstance(target_ratio, tuple):
            target_ratio = (target_ratio[0], target_ratio[0] / 1e5) \
                if len(target_ratio) == 1 else target_ratio
        elif isinstance(target_ratio, list):
            target_ratio = (target_ratio[0], target_ratio[0] / 1e5) \
                if len(target_ratio) == 1 else target_ratio
        else:
            raise ValueError('target_ratio should be either float '
                             f'or tuple (list), got {type(target_ratio)}')

        assert len(target_ratio) == 2, \
            '"target_ratio" must be list or tuple of two floats'
        assert 0 <= step_ratio_up < 1.0, \
            '"step_ratio_up" must be in range [0,1)'

        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up
        self.lr_phases = []  # init lr_phases

        assert not by_epoch, \
            'currently only support "by_epoch" = False'
        super(CyclicLrAdditionalHook, self).__init__(**kwargs)

    def before_run(self, runner):
        super(CyclicLrAdditionalHook, self).before_run(runner)
        # initiate lr_phases
        # total lr_phases are separated as up and down
        max_iter_per_phase = runner.max_iters // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * max_iter_per_phase)
        self.lr_phases.append(
            [0, iter_up_phase, max_iter_per_phase, 1, self.target_ratio[0]])
        self.lr_phases.append([
            iter_up_phase, max_iter_per_phase, max_iter_per_phase,
            self.target_ratio[0], self.target_ratio[1]
        ])

    def get_lr(self, runner, base_lr):
        curr_iter = runner.iter
        for (start_iter, end_iter, max_iter_per_phase, start_ratio,
             end_ratio) in self.lr_phases:
            curr_iter %= max_iter_per_phase
            if start_iter <= curr_iter < end_iter:
                progress = curr_iter - start_iter
                return annealing_cos(base_lr * start_ratio,
                                     base_lr * end_ratio,
                                     progress / (end_iter - start_iter))


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.

    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


class CustomSchedulerHook(Hook):
    """Custom Scheduler Hook.

    Args:
        attr_name (str): Name of the attribute
        attr_base (float): The initial value of the attribute
        by_epoch (bool): Attr changes epoch by epoch. If by_epoch is True, the
            attribute will be updated each epoch when warm_up is none. You can
            set by_epoch to false (i.e., by iter) for more frequently update.
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): Attr used at the beginning of warmup equals to
            warmup_ratio * initial_attr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(self,
                attr_name="",
                attr_base=None,
                by_epoch=True,
                warmup=None,
                warmup_iters=0,
                warmup_ratio=0.1,
                warmup_by_epoch=False,
                update_interval=1,
                **kwargs):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 <= warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range [0,1]'
        
        # basic custom scheduler args
        self.attr_name = attr_name
        self.attr_base = attr_base  # initial attr for optinal param groups
        if attr_base is None or attr_name == "":
            raise ValueError(
                f'invalid attr_name="{attr_name}" or attr_base="{attr_base}"')
        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch
        self.update_interval = update_interval

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.regular_attr = attr_base  # expected attr if no warming up is performed

    def _set_attr(self, runner, attr):
        setattr(runner.model.module, self.attr_name, attr)

    def get_attr(self, runner, base_attr):
        raise NotImplementedError

    def get_regular_attr(self, runner):
        return self.get_attr(
            runner, getattr(runner.model.module, self.attr_name))

    def get_warmup_attr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_attr = self.warmup_ratio * self.attr_base
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_attr = (1 - k) * self.attr_base
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_attr = k * self.attr_base
        return warmup_attr

    def before_run(self, runner):
        # notice: we can only get the attr_name of the model.module attribute,
        # but we cannot adjust the attribute of submodele, such as the attribute
        # in runner.model.head.
        assert hasattr(runner.model.module, self.attr_name), \
                "The runner must have attribute:"+self.attr_name
        attr = getattr(runner.model.module, self.attr_name)
        assert isinstance(attr, (float, int))
        if self.attr_base != attr and self.warmup is None:
            print_log(f"CustomSchedulerHook: overwrite {self.attr_name}={self.attr_base}.")
        setattr(runner.model.module, self.attr_name, self.attr_base)

    def before_train_epoch(self, runner):
        if self.warmup_by_epoch:
            epoch_len = len(runner.data_loader)
            self.warmup_iters = self.warmup_epochs * epoch_len
        if not self.by_epoch:
            return
        # self.regular_attr = self.get_regular_attr(runner)
        self._set_attr(runner, self.regular_attr)

    def before_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            if not self.by_epoch:
                if self.warmup is None or cur_iter > self.warmup_iters:
                    # using get_regular_attr() after finishing the warmup stage
                    self.regular_attr = self.get_regular_attr(runner)
                    self._set_attr(runner, self.regular_attr)
                else:
                    warmup_attr = self.get_warmup_attr(cur_iter)
                    self._set_attr(runner, warmup_attr)
            elif self.by_epoch:
                if self.warmup is None or cur_iter > self.warmup_iters:
                    return
                elif cur_iter == self.warmup_iters:
                    self._set_attr(runner, self.regular_attr)
                else:
                    warmup_attr = self.get_warmup_attr(cur_iter)
                    self._set_attr(runner, warmup_attr)


@HOOKS.register_module()
class CustomFixedHook(CustomSchedulerHook):

    def __init__(self, **kwargs):
        super(CustomFixedHook, self).__init__(**kwargs)

    def get_attr(self, runner, base_attr):
        return base_attr


@HOOKS.register_module()
class CustomStepHook(CustomSchedulerHook):

    def __init__(self, step, gamma=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        super(CustomStepHook, self).__init__(**kwargs)

    def get_attr(self, runner, base_attr):
        progress = runner.epoch if self.by_epoch else runner.iter

        if isinstance(self.step, int):
            return base_attr * (self.gamma**(progress // self.step))

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        return base_attr * self.gamma**exp


@HOOKS.register_module()
class CustomExpHook(CustomSchedulerHook):

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(CustomExpHook, self).__init__(**kwargs)

    def get_attr(self, runner, base_attr):
        progress = runner.epoch if self.by_epoch else runner.iter
        return base_attr * self.gamma**progress


@HOOKS.register_module()
class CustomPolyHook(CustomSchedulerHook):

    def __init__(self, power=1., min_attr=0., **kwargs):
        self.power = power
        self.min_attr = min_attr
        super(CustomPolyHook, self).__init__(**kwargs)

    def get_attr(self, runner, base_attr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_attr - self.min_attr) * coeff + self.min_attr


@HOOKS.register_module()
class CustomCosineAnnealingHook(CustomSchedulerHook):

    def __init__(self, min_attr=None, min_attr_ratio=None, **kwargs):
        assert (min_attr is None) ^ (min_attr_ratio is None)
        self.min_attr = min_attr
        self.min_attr_ratio = min_attr_ratio
        super(CustomCosineAnnealingHook, self).__init__(**kwargs)

    def get_attr(self, runner, base_attr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        if self.min_attr_ratio is not None:
            target_attr = base_attr * self.min_attr_ratio
        else:
            target_attr = self.min_attr
        return annealing_cos(base_attr, target_attr, progress / max_progress)
