from openmixup.utils import ConfigGenerator


def main():
    """Automatic Config Generator: generate openmixup configs in terms of keys

    Usage:
        Generating various mixup methods' configs by executing
            `python configs/classification/aircrafts/mixups/auto_train_mixups.py`
    """

    # *** mixups ***
    base_path = "configs/classification/aircrafts/mixups/basic/r18_mixups_CE_none.py"
    # base_path = "configs/classification/aircrafts/mixups/basic/rx50_mixups_CE_none.py"

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
        'runner.max_epochs': [200,],
    }
    
    num_device = 1
    
    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()