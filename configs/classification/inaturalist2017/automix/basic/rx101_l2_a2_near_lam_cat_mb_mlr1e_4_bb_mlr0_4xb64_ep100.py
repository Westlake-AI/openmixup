_base_ = "r50_l2_a2_near_lam_cat_mb_mlr1e_4_bb_mlr0_4xb64_ep100.py"

# model settings
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        num_stages=4,
        groups=32, width_per_group=4,  # 32x4d
        out_indices=(2,3),  # stage-3 for MixBlock, x-1: stage-x
        style='pytorch'),
)
