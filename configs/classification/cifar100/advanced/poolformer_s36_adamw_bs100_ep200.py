_base_ = [
    '../../_base_/datasets/cifar100/sz32_swin_bs100.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[1, 0.8],
    mix_mode=['cutmix', 'mixup'],
    backbone=dict(
        type='PoolFormer',
        arch='s36',
        drop_path_rate=0.2,
        in_patch_size=3, in_stride=1, in_pad=1,  # CIFAR version
    ),
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=100, mode='original', loss_weight=1.0),
        with_avg_pool=True,
        in_channels=512, num_classes=100),
    init_cfg=[
        dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02, bias=0.),
        dict(type='Constant', layer=['LayerNorm', 'GroupNorm'], val=1., bias=0.)
    ],
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'layer_scale': dict(weight_decay=0.),
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
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
