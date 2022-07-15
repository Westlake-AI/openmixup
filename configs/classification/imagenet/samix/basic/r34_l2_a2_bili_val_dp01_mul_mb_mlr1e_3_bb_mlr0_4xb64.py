_base_ = "r18_l2_a2_bili_val_dp01_mul_mb_mlr1e_3_bb_mlr0_4xb64.py"

# model settings
model = dict(
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(2,3),  # stage-3 for MixBlock, x-1: stage-x
        style='pytorch'),
)
