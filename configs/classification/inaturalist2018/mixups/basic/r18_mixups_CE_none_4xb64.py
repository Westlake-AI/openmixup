_base_ = "r50_mixups_CE_none_4xb64.py"

# model settings
model = dict(
    backbone=dict(
        type='ResNet',  # normal
        # type='ResNet_Mix',  # required by 'manifoldmix'
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style='pytorch'),
    head=dict(
        type='ClsHead',  # mixup head, normal CE loss
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=8142)
)
