_base_ = "resnet50_4xb64_step_ep100.py"

# model settings
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=154,
        groups=32, width_per_group=4,  # 32x4d
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style='pytorch'),
)
