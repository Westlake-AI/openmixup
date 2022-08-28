from openmixup.utils import ConfigGenerator


def main():
    """Automatic Config Generator: generate openmixup configs in terms of keys

    Usage:
        Generating various mixup methods' configs by executing
            `python configs/classification/inaturalist2017/auto_train_mixups.py`
    """

    # *** mixups ***
    base_path = "configs/classification/inaturalist2017/mixups/r18_mixups_CE_none_4xb64.py"
    # base_path = "configs/classification/inaturalist2017/mixups/r50_mixups_CE_none_4xb64.py"
    # base_path = "configs/classification/inaturalist2017/mixups/rx101_mixups_CE_none_4xb64.py"

    # abbreviation of long attributes
    abbs = {
        'max_epochs': 'ep'
    }
    # create nested dirs (cannot be none)
    model_var = {
        'model.mix_mode': ["mixup", "cutmix",],
        # 'model.mix_mode': ["vanilla", "mixup", "cutmix", "manifoldmix", "fmix", "saliencymix", "resizemix", "puzzlemix",],
    }
    # adjust sub-attributes (cannot be none)
    gm_var = {
        'model.alpha': [0.2, 1,],  # default: 0.2
        'runner.max_epochs': [100,],
    }
    
    num_device = 4  # 4 gpus by default
    
    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()