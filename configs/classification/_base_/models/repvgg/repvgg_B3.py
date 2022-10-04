# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=0.2,
    mix_mode="mixup",
    mix_args=dict(),
    backbone=dict(
        type='RepVGG',
        arch='B3',
        out_indices=(3,)
    ),
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='classy_vision', loss_weight=1.0),
        with_avg_pool=True,
        in_channels=2560, num_classes=1000),
    init_cfg=[
        dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02, bias=0.),
        dict(type='Constant', layer='BatchNorm', val=1., bias=0.)
    ],
)
