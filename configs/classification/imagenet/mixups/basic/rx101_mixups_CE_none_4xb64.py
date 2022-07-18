_base_ = "r50_mixups_CE_none_4xb64.py"

# model settings
model = dict(
    backbone=dict(
        # type='ResNeXt',  # normal
        type='ResNeXt_Mix',  # required by 'manifoldmix'
        depth=101,
        groups=32, width_per_group=4,  # 32x4d
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style='pytorch'),
)
