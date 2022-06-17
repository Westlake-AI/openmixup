# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='VGG',
        depth=16, num_classes=1000,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=False, num_classes=None)
)
