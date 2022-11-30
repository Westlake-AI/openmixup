# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(
        type='MogaNet',
        arch="small",
        init_value=1e-5,
        drop_path_rate=0.1,
        stem_norm_cfg=dict(type='BN', eps=1e-5),
        conv_norm_cfg=dict(type='BN', eps=1e-5),
        attn_force_fp32=True,  # force fp32 of gating for fp16 training
    ),
    head=dict(
        type='ClsMixupHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=True,
        in_channels=512, num_classes=1000,
    ),
    init_cfg=[
        dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02, bias=0.),
        dict(type='Constant', layer=['BatchNorm', 'LayerNorm'], val=1., bias=0.)
    ],
)
