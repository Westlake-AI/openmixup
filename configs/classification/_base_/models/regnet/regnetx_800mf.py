# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='RegNet',
        arch='regnetx_800mf',
        out_indices=(3,)
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=672, num_classes=1000)
)
