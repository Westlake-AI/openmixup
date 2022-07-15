_base_ = "r18_l2_a2_near_lam_cat_mb_mlr1e_3_bb_mlr0.py"

# model settings
model = dict(
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(2,3),  # stage-3 for MixBlock, x-1: stage-x
        style='pytorch'),
)
