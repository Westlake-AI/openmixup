_base_ = [
    '../../_base_/datasets/cifar100/sz224_swin_bs100.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(
        type='PCPVT',
        arch='small',
        in_channels=3,
        out_indices=(3, ),
        qkv_bias=True,
        norm_cfg=dict(type='LN', eps=1e-06),
        norm_after_stage=[False, False, False, True],
        drop_rate=0.0,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
    ),
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=100, mode='original', loss_weight=1.0),
        with_avg_pool=True, multi_label=True, in_channels=512, num_classes=100,
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-4,
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'absolute_pos_embed': dict(weight_decay=0.),
        'relative_position_bias_table': dict(weight_decay=0.),
    })

# interval for accumulate gradient
update_interval = 1  # total: 1 x bs100 x 1 accumulates = bs100

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-6,
    warmup='linear',
    warmup_iters=20, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
