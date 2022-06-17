# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='EfficientNet',
        arch='b6',
        out_indices=(6,),  # x-1: stage-x
        norm_cfg=dict(type='BN', eps=1e-3),
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=2304, num_classes=1000)
)
