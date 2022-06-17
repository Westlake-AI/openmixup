# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='ShuffleNetV2', widen_factor=1.0,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=1024, num_classes=1000)
)
