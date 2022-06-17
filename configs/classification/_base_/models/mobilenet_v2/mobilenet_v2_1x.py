# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,
        out_indices=(7,),  # x-1: stage-x
        norm_cfg=dict(type='BN'),
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=1280, num_classes=1000)
)
