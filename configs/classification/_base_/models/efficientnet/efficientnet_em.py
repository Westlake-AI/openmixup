# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='EfficientNet',
        arch='em',  # `em` means EfficientNet-EdgeTPU-M arch
        out_indices=(6,),  # x-1: stage-x
        norm_cfg=dict(type='BN', eps=1e-3),
        act_cfg=dict(type='ReLU'),
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=1024, num_classes=1000)
)
