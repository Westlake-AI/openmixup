# model settings
model = dict(
    type='Classification',
    backbone=dict(type='HRNet', arch='w44'),
    neck=dict(
        type='HRFuseScales',
        in_channels=(44, 88, 176, 352)),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=2048, num_classes=1000)
)
