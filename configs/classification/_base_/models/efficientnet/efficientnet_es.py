# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='EfficientNet',
        arch='es',  # `es` means EfficientNet-EdgeTPU-S arch
        out_indices=(6,),  # x-1: stage-x
        norm_cfg=dict(type='BN', eps=1e-3),
        act_cfg=dict(type='ReLU'),
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=1280, num_classes=1000)
)
