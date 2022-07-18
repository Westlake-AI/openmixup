_base_ = "r50_mixups_CE_none_4xb64.py"

# model settings
model = dict(
    backbone=dict(
        # type='ResNet',  # normal
        type='ResNet_Mix',  # required by 'manifoldmix'
        depth=101,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style='pytorch'),
)
