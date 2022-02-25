from openmixup.utils import ConfigGenerator


def main():
    """Automatic Config Generator: generate openmixup configs in terms of keys

    Usage:
        Generating various mixup methods' configs by executing
            `python configs/classification/imagenet/mixups/auto_train_in_mixups.py`
        For example: generate the optimal configs for 'mixup' with 'default CE' on
            ImageNet-1k based on R-18 as following folders:
            |-- configs/classification/imagenet/mixups/basic/
            |   |--> r18/mix_modemixup/
            |   |   |--> r18_mixups_CE_none_alpha0_2_ep100.json
            |   |   |--> r18_mixups_CE_none_alpha0_2_ep300.json
            |   |   |--> ...
    """

    # *** default CE ***
    base_path = "configs/classification/imagenet/mixups/basic/r18_mixups_CE_none.py"
    # base_path = "configs/classification/imagenet/mixups/basic/r34_mixups_CE_none.py"
    # base_path = "configs/classification/imagenet/mixups/basic/r50_mixups_CE_none.py"
    # base_path = "configs/classification/imagenet/mixups/basic/r101_mixups_CE_none.py"
    # base_path = "configs/classification/imagenet/mixups/basic/rx101_mixups_CE_none.py"

    # *** soft CE ***
    # base_path = "configs/classification/imagenet/mixups/basic/r50_mixups_CE_soft.py"

    # *** BCE (sigmoid) ***
    # base_path = "configs/classification/imagenet/mixups/basic/r50_mixups_CE_sigm.py"

    # *** multi-mode mixup (using various mixup policies) ***
    # base_path = "configs/classification/imagenet/mixups/basic/r50_mixups_CE_soft_multi_mode.py"
    # base_path = "configs/classification/imagenet/mixups/rbs_a3/r50_rsb_a3_CE_sigm_multi_mode_sz160_bs2048.py"

    # *** decouple mixup ***
    # base_path = "configs/classification/imagenet/mixups/decouple/r50_mixups_CE_soft_decouple.py"

    # *** DeiT (Swim) ***
    # base_path = "configs/classification/imagenet/mixups/deit/deit_s_timm_smooth_mix0_8_cut1_0_sz224_bs1024_ema.py"

    # *** RSB A2 ***
    # base_path = "configs/classification/imagenet/mixups/rsb_a2/r50_rsb_a2_CE_sigm_mix0_1_cut1_0_sz224_bs2048_fp16.py"

    # *** RSB A3 ***
    # base_path = "configs/classification/imagenet/mixups/rsb_a3/r50_rsb_a3_CE_sigm_sz160_bs2048_fp16.py"

    # abbreviation of long attributes
    abbs = {
        'total_epochs': 'ep'
    }
    # create nested dirs (cannot be none)
    model_var = {
        'model.mix_mode': ["mixup", "cutmix",],
        # 'model.mix_mode': ["vanilla", "mixup", "cutmix", "manifoldmix", "fmix", "saliencymix", "resizemix",],
    }
    # adjust sub-attributes (cannot be none)
    gm_var = {
        'model.alpha': [0.2, 1,],  # default: 1
        # 'model.head.loss.use_soft': [True, ],
        # 'model.head.loss.use_sigmoid': [True, ],
        # 'lr_config.min_lr': [0],  # default: 0
        'total_epochs': [100, 300],
    }
    
    num_device = 1
    
    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()