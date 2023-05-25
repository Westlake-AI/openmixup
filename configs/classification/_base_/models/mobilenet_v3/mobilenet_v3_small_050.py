# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='MobileNetV3',
        arch='small_050',
        norm_cfg=dict(type='BN', eps=1e-5, momentum=0.1),
    ),
    head=dict(
        type='StackedLinearClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        in_channels=288, num_classes=1000, mid_channels=[1024],
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
        with_avg_pool=True)
)
