from openmixup.utils import ConfigGenerator


def main():
    """Automatic Config Generator: generate openmixup configs in terms of keys

    Usage:
        Generating configs for AutoMix by executing
            `python configs/classification/cifar100/automix/auto_train_cifar_automix.py`
        For example: generate the optimal configs for 'AutoMix' with 'default CE' on
            CIFAR-100 based on R-18 using 'nearest' upsampling as following:
            |-- configs/classification/cifar100/automix/basic/
            |   |--> r18/unsampling_modenearest/
            |   |   |--> r18_l2_a2_near_lam_cat_L1_01_mlr5e_2_ep800.py
            |   |   |--> ...
    """

    ## basic
    base_path = "configs/classification/cifar100/samix/basic/r18_l2_a2_bili_val_dp0_mul_x_cat_L1_var_01_mlr5e_2.py"
    # base_path = "configs/classification/cifar100/samix/basic/rx50_l2_a2_bili_val_dp0_mul_x_cat_L1_var_01_mlr5e_2.py"
    # base_path = "configs/classification/cifar100/samix/basic/wrn28_8_l1_a2_bili_val_dp0_mul_x_cat_L1_var_01_mlr1e_3.py"

    # abbreviation of long attributes
    abbs = {
        'total_epochs': 'ep'
    }
    # create nested dirs (cannot be none)
    model_var = {
        'model.mix_block.unsampling_mode': ['bilinear',],
    }
    # adjust sub-attributes (cannot be none)
    gm_var = {
        # 'model.alpha': [2,],  # default: 2
        # 'model.head_mix.loss.use_soft': [True, ],  # soft CE for bb cls
        # 'model.head_mix.loss.use_sigmoid': [True, ],  # BCE for bb cls
        # 'model.mask_adjust': [0, 0.25, 0.50,],  # SAMix, small datasets, default: 0
        # 'model.mix_block.lam_mul_k': [-1, 0.25, 0.5, 1],  # SAMix, small datasets, default: -1
        # # 'lr_config.min_lr': [5e-2, 1e-3, 1e-4, 0],  # AutoMix default: 5e-2
        'total_epochs': [400, 800, 1200]
    }
    
    num_device = 1
    
    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()