# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='MobileNetV3',
        arch='large',
        out_indices=(16,),  # x-1: stage-x
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=960, num_classes=1000)
)
