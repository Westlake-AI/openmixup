# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(
        type='UniFormer',
        arch='base',
        drop_path_rate=0.3,
        init_values=1e-6,
        init_cfg=[
            dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02, bias=0.),
            dict(type='Constant', layer=['BatchNorm', 'LayerNorm'], val=1., bias=0.)
        ],
    ),
    head=dict(
        type='ClsMixupHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=True,
        in_channels=512, num_classes=1000,
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        ]),
)
