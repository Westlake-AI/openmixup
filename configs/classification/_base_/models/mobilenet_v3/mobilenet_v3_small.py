# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='MobileNetV3',
        arch='small',
    ),
    head=dict(
        type='StackedLinearClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        in_channels=576, num_classes=1000, mid_channels=[1024],
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
        with_avg_pool=True)
)
