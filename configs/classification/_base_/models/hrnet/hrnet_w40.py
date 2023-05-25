# model settings
model = dict(
    type='Classification',
    backbone=dict(type='HRNet', arch='w40'),
    neck=dict(
        type='HRFuseScales',
        in_channels=(40, 80, 160, 320)),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=2048, num_classes=1000)
)
