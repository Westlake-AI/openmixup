# model settings
model = dict(
    type='DeepCluster',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='BN'),
        style='pytorch'),
    neck=dict(type='AvgPoolNeck'),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=False,  # already has avgpool in the neck
        in_channels=512, num_classes=10000)
)
