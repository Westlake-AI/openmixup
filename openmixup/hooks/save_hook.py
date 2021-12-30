import os
from mmcv.runner import Hook

from .registry import HOOKS


@HOOKS.register_module
class SAVEHook(Hook):
    """Hook for saving.

    Args:
        save_interval (float): Default: 1.
        iter_per_epoch (int): The iter number of each epoch.
    """

    def __init__(self, save_interval=1., iter_per_epoch=500, update_interval=1, **kwargs):
        self.save_interval = save_interval
        self.iter_per_epoch = iter_per_epoch
        self.update_interval = update_interval

    def before_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            if cur_iter % self.save_interval == 0:
                runner.model.module.save = True
                runner.model.module.save_name = '{}/epoch_{}.png'.format(
                    runner.work_dir, int(cur_iter/self.iter_per_epoch))
                save_name = os.path.join(runner.work_dir, "MixedSamples")
                if not os.path.exists(save_name):
                    try:
                        os.mkdir(save_name)
                    except:
                        if not os.path.exists(save_name):
                            save_name = runner.work_dir
                        print("mkdir error")
                runner.model.module.save_name = os.path.join(
                    save_name, "epoch_{}.png".format(int(cur_iter/self.iter_per_epoch)))
            else:
                runner.model.module.save = False