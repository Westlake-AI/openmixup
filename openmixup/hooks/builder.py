import re
import torch.distributed as dist

from openmixup.utils import build_from_cfg, print_log
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
