import re
import torch.distributed as dist
from openmixup.utils import build_from_cfg, optimizers, print_log

from mmcv.runner import obj_from_dict

from .registry import HOOKS


def build_hook(cfg, default_args=None):
    return build_from_cfg(cfg, HOOKS, default_args)


def build_addtional_scheduler(param_names, hook_cfg):
    """Build Addtional Scheduler from configs.

    Args:
        param_names (list): Names of parameters in the model.
        hook_cfg (dict): The config dict of the optimizer.

    Returns:
        obj: The constructed object.
    """
    hook_cfg = hook_cfg.copy()
    paramwise_options = hook_cfg.pop('paramwise_options', None)
    # you must use paramwise_options in optimizer_cfg
    assert isinstance(paramwise_options, list)
    addtional_indice = list()
    for i, name in enumerate(param_names):
        for regexp in paramwise_options:
            if re.search(regexp, name):
                # additional scheduler for selected params
                addtional_indice.append(i)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print_log('optional_scheduler -- {}: {}'.format(name, 'lr'))
    # build type
    assert 'policy' in hook_cfg
    policy_type = hook_cfg.pop('policy')
    # If the type of policy is all in lower case
    if policy_type == policy_type.lower():
        policy_type = policy_type.title()
    hook_cfg['type'] = policy_type + 'LrAdditionalHook'
    # fatal args
    hook_cfg['addtional_indice'] = addtional_indice
    return build_hook(hook_cfg, dict(dist_mode=True))


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with regular expression as keys
                  to match parameter names and a dict containing options as
                  values. Options include 6 fields: lr, lr_mult, momentum,
                  momentum_mult, weight_decay, weight_decay_mult.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> paramwise_options = {
        >>>     '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay_mult=0.1),
        >>>     '\Ahead.': dict(lr_mult=10, momentum=0)}
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001,
        >>>                      paramwise_options=paramwise_options)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, optimizers,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue

            for regexp, options in paramwise_options.items():
                if re.search(regexp, name):
                    for key, value in options.items():
                        if key.endswith('_mult'): # is a multiplier
                            key = key[:-5]
                            assert key in optimizer_cfg, \
                                "{} not in optimizer_cfg".format(key)
                            value = optimizer_cfg[key] * value
                        param_group[key] = value
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            print_log('paramwise_options -- {}: {}={}'.format(
                                name, key, value))

            # otherwise use the global settings
            params.append(param_group)

        optimizer_cls = getattr(optimizers, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)
