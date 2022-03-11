from openmixup.utils import ConfigGenerator


def main():
    # Please generate SAMix configs by auto_train.py

    # *** default CE ***
    base_path = "configs/classification/imagenet/samix/basic/r18_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0.py"
    # base_path = "configs/classification/imagenet/samix/basic/r34_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0.py"
    # base_path = "configs/classification/imagenet/samix/basic/r50_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0.py"
    # base_path = "configs/classification/imagenet/samix/basic/r101_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0.py"
    # base_path = "configs/classification/imagenet/samix/basic/rx101_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0.py"

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
        # 'addtional_scheduler.min_lr': [1e-3, 1e-4, 0],  # AutoMix default: 1e-3
        'total_epochs': [100, 300]
    }
    
    num_device = 1
    
    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()