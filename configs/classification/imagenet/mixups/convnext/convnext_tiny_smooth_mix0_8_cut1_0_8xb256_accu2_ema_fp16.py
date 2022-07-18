_base_ = [
    '../../../_base_/datasets/imagenet/swin_sz224_4xbs256.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(
        type='ConvNeXt',
        arch='tiny',
        out_indices=(3,),
        norm_cfg=dict(type='LN2d', eps=1e-6),
        act_cfg=dict(type='GELU'),
        drop_path_rate=0.1,
        gap_before_final_norm=True,
    ),
    head=dict(
        type='ClsMixupHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=False,  # gap_before_final_norm is True
        in_channels=768, num_classes=1000)
)

# interval for accumulate gradient
update_interval = 2  # total: 8 x bs256 x 2 accumulates = bs4096

# additional hooks
custom_hooks = [
    dict(type='EMAHook',  # EMA_W = (1 - m) * EMA_W + m * W
        momentum=0.9999,
        warmup='linear',
        warmup_iters=20 * 626, warmup_ratio=0.9,  # warmup 20 epochs.
        update_interval=update_interval,
    ),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=4e-3,  # lr = 5e-4 * (256 * 4) * 4 accumulate / 1024 = 4e-3 / bs4096
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),
    })

# apex
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=None, update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-6,
    warmup='linear',
    warmup_iters=20, warmup_by_epoch=True,  # warmup 20 epochs.
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
