# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='ShuffleNetV1', groups=3, widen_factor=1.0,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=960, num_classes=1000)
)
