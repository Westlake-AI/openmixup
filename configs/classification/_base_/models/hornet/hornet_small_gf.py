# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(
        type='HorNet',
        arch='small-gf',
        drop_path_rate=0.4,
    ),
    head=dict(
        type='ClsMixupHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=True,
        in_channels=768, num_classes=1000),
    init_cfg=[
        dict(type='TruncNormal', layer=['Linear'], std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
        dict(type='Constant', layer=['LayerScale'], val=1e-6),
    ],
)
