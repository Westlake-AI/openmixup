# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(
        type='ConvNeXt',
        arch='large',
        out_indices=(3,),  # x-1: stage-x
        act_cfg=dict(type='GELU'),
        drop_path_rate=0.5,
        gap_before_final_norm=True,
        init_cfg=[
            dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ],
    ),
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=False,
        in_channels=1536, num_classes=1000,
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        ]),
)
