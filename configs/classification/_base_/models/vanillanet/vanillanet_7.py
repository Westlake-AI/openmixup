# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.4, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    cosine_update = True,  # update act_learn in the backbone
    backbone=dict(
        type='VanillaNet',
        arch='vanillanet_7',
        act_learn_init=1.0,
        act_learn_invert=True,
    ),
    head=dict(
        type='VanillaNetClsHead',  # mlp head + BCE loss
        loss=dict(type='CrossEntropyLoss',  # mixup BCE loss (one-hot encoding)
            use_soft=False, use_sigmoid=True, loss_weight=1.0),
        with_avg_pool=False, multi_label=True, two_hot=False,
        drop_rate=0.05,
        in_channels=1024*4, num_classes=1000),
    init_cfg=[
        dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02, bias=0.),
        dict(type='Constant', layer='BatchNorm', val=1., bias=0.)
    ],
)
