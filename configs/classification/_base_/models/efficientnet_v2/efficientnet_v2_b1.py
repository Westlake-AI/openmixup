# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='EfficientNetV2',
        arch='b1',
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=1280, num_classes=1000)
)
