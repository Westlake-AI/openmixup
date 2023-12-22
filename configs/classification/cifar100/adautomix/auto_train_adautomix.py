from openmixup.utils import ConfigGenerator


def main():

    # *** default CE ***
    base_path = "configs/classification/cifar100/adautomix/basic/r18_l2_a1_bili_mlr5e_2.py"

    # abbreviation of long attributes
    abbs = {
        'total_epochs': 'ep'
    }
    # create nested dirs (cannot be none)
    model_var = {
        'models.mix_block.unsampling_mode': 'nearest',
    }
    # adjust sub-attributes (cannot be none)
    gm_var = {
        # 'models.alpha': [2,],  # default: 2
        # 'models.head_mix.loss.use_soft': [True, ],  # soft CE for bb cls
        # 'models.head_mix.loss.use_sigmoid': [True, ],  # BCE for bb cls
        # # 'lr_config.min_lr': [5e-2, 1e-3, 1e-4, 0],  # AutoMix default: 5e-2
        'total_epochs': [400, 800, 1200]
    }

    num_device = 1

    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()