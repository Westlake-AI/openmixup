# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(
        type='T2T_ViT',
        img_size=224,
        in_channels=3,
        embed_dims=384,
        t2t_cfg=dict(
            token_dims=64,
            use_performer=False,
        ),
        num_layers=14,
        layer_cfgs=dict(
            num_heads=6,
            feedforward_channels=3 * 384,  # mlp_ratio = 3
        ),
        drop_path_rate=0.1,
    ),
    head=dict(
        type='VisionTransformerClsHead',
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        in_channels=384, num_classes=1000),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
)
