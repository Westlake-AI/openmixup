_base_ = [
    '../../_base_/models/starnet/starnet_s4.py',
    '../../_base_/datasets/imagenet/starnet_sz224_8xbs256.py',
    '../../_base_/default_runtime.py',
]

# additional hooks
update_interval = 1  # 256 x 8gpus x 1 accumulates = bs2048

# optimizer
optimizer = dict(
    type='AdamW',
    lr=3e-3,
    weight_decay=0.025, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
