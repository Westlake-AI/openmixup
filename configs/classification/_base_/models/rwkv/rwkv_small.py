# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(
        type='RWKV',
        arch='rwkv_small',
        img_size=224,
        drop_path_rate=0.2,
        layer_scale_init=1e-6,
        gap_before_final_norm=True,
    ),
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=False,
        in_channels=512, num_classes=1000),
)
