# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(
        type='DeiT3',
        arch='large',
        img_size=192,
        patch_size=16,
        drop_path_rate=0.45),
    head=dict(
        type='VisionTransformerClsHead',
        loss=dict(type='CrossEntropyLoss',  # mixup BCE loss (one-hot encoding)
            use_soft=False, use_sigmoid=True, loss_weight=1.0),
        multi_label=True,
        in_channels=1024, num_classes=1000),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
)
