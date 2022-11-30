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
    base_path = "configs/classification/imagenet/mixups/basic/r18_mixups_CE_none_4xb64.py"
    # base_path = "configs/classification/imagenet/mixups/basic/r34_mixups_CE_none_4xb64.py"
    # base_path = "configs/classification/imagenet/mixups/basic/r50_mixups_CE_none_4xb64.py"
    # base_path = "configs/classification/imagenet/mixups/basic/r101_mixups_CE_none_4xb64.py"
    # base_path = "configs/classification/imagenet/mixups/basic/rx101_mixups_CE_none_4xb64.py"

    # *** soft CE ***
    # base_path = "configs/classification/imagenet/mixups/basic/r50_mixups_CE_soft_4xb64.py"

    # *** BCE (sigmoid) ***
    # base_path = "configs/classification/imagenet/mixups/basic/r50_mixups_CE_sigm_4xb64.py"

    # *** multi-mode mixup (using various mixup policies) ***
    # base_path = "configs/classification/imagenet/mixups/basic/r50_mixups_CE_soft_multi_mode_4xb64.py"
    # base_path = "configs/classification/imagenet/mixups/rbs_a3/r50_rsb_a3_CE_sigm_multi_mode_4xb512.py"

    # *** decouple mixup ***
    # base_path = "configs/classification/imagenet/mixups/decouple/r50_mixups_CE_soft_decouple_4xb64.py"

    # *** DeiT, PVT, Swin ***
    # base_path = "configs/classification/imagenet/mixups/deit/deit_tiny_smooth_mix_4xb256.py"
    # base_path = "configs/classification/imagenet/mixups/deit/deit_small_smooth_mix_8xb128.py"
    # base_path = "configs/classification/imagenet/mixups/pvt/pvt_tiny_smooth_mix_4xb256.py"
    # base_path = "configs/classification/imagenet/mixups/swin/swin_tiny_smooth_mix_4xb256_fp16.py"

    # *** RSB A2 ***
    # base_path = "configs/classification/imagenet/mixups/rsb_a2/r50_rsb_a2_CE_sigm_mix0_1_cut1_0_4xb256_accu2_fp16.py"

    # *** RSB A3 ***
    # base_path = "configs/classification/imagenet/mixups/rsb_a3/r50_rsb_a3_CE_sigm_4xb512_fp16.py"

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
        'model.alpha': [0.2, 1,],  # default: 1
        # 'model.head.loss.use_soft': [True, ],
        # 'model.head.loss.use_sigmoid': [True, ],
        # 'lr_config.min_lr': [0],  # default: 0
        'runner.max_epochs': [100, 300,],
    }
    
    num_device = 4  # 4 gpus by default
    
    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()