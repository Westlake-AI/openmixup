from openmixup.utils import ConfigGenerator


def main():
    """Automatic Config Generator: generate openmixup configs in terms of keys

    Usage:
        Generating various mixup methods' configs by executing
            `python configs/classification/tiny_imagenet/mixups/auto_train_mixups.py`
        For example: generate the optimal configs for 'mixup' with 'default CE' on
            Tiny-ImageNet based on R-18 as following folders:
            |-- configs/classification/tiny_imagenet/mixups/basic/
            |   |--> r18/mix_modemixup/
            |   |   |--> r18_mixups_CE_none_alpha1_ep800.py
            |   |   |--> ...
    """

    # *** default CE ***
    base_path = "configs/classification/tiny_imagenet/mixups/baseline/r18_mixups_CE_none.py"
    # base_path = "configs/classification/tiny_imagenet/mixups/basic/rx50_mixups_CE_none.py"
    # base_path = "configs/classification/tiny_imagenet/mixups/basic/wrn28_8_mixups_CE_none.py"
    
    # *** soft CE ***
    # base_path = "configs/classification/tiny_imagenet/mixups/basic/r18_mixups_CE_soft.py"
    
    # *** BCE (sigmoid) ***
    # base_path = "configs/classification/tiny_imagenet/mixups/basic/r18_mixups_CE_sigm.py"
    
    # *** multi-mode mixup (using various mixup policies) ***
    # base_path = "configs/classification/tiny_imagenet/mixups/basic/r18_mixups_CE_none_multi_mode.py"
    
    # *** decouple mixup ***
    # base_path = "configs/classification/tiny_imagenet/mixups/decouple/r18_mixups_CE_soft_decouple.py"

    # abbreviation of long attributes
    abbs = {
        'max_epochs': 'ep'
    }
    # create nested dirs (cannot be none)
    model_var = {
        'model.mix_mode': ["mixup",],
        # 'model.mix_mode': ["vanilla", "mixup", "cutmix", "manifoldmix", "fmix", "saliencymix", "resizemix", "puzzlemix",],
    }
    # adjust sub-attributes (cannot be none)
    gm_var = {
        'model.alpha': [0.2, 1,],  # default: 1
        # 'model.head.loss.use_soft': [True, ],
        # 'model.head.loss.use_sigmoid': [True, ],
        # 'optimizer.weight_decay': [1e-4, 5e-4, 1e-3],  # default: 1e-4, adjust for RX50 and WRN
        'runner.max_epochs': [400,],
        # 'runner.max_epochs': [400, 800, 1200,],
    }
    
    num_device = 1
    
    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()