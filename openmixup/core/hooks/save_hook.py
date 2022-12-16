import os
import mmcv
from mmcv.runner import Hook

from .registry import HOOKS


@HOOKS.register_module
class SAVEHook(Hook):
    """Hook for saving.

    Args:
        suffix (str): File suffix in {'png', 'pdf'}.
        save_interval (float): Every iter or epoch to save. Default: 1.
        iter_per_epoch (int): The iter number of each epoch.
    """

    def __init__(self,
                 suffix='png',
                 save_interval=1,
                 iter_per_epoch=500,
                 update_interval=1,
                 **kwargs):
        self.suffix = suffix
        self.save_interval = save_interval
        self.iter_per_epoch = iter_per_epoch
        self.update_interval = update_interval
        self.save_dir = ""

    def before_run(self, runner):
        save_name = runner.model.module.save_name
        self.save_dir = os.path.join(runner.work_dir, save_name)
        mmcv.mkdir_or_exist(self.save_dir)

    def before_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            if cur_iter % self.save_interval == 0:
                runner.model.module.save = True
                runner.model.module.save_name = os.path.join(
                    self.save_dir, "epoch_{}.{}".format(int(cur_iter/self.iter_per_epoch), self.suffix))
            else:
                runner.model.module.save = False
