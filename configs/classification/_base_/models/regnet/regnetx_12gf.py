# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='RegNet',
        arch='regnetx_12gf',
        out_indices=(3,)
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=2240, num_classes=1000)
)
