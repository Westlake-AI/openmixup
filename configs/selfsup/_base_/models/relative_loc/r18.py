# model settings
model = dict(
    type='RelativeLoc',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='BN'),
        style='pytorch'),
    neck=dict(
        type='RelativeLocNeck',
        in_channels=512, out_channels=4096,
        with_avg_pool=True),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=False,  # already has avgpool in the neck
        in_channels=4096, num_classes=8),
)
