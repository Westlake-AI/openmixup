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
        type='StackedLinearClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        in_channels=960, num_classes=1000, mid_channels=[1280],
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
        with_avg_pool=True)
)
