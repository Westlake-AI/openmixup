_base_ = [
    '../_base_/models/r50.py',
    '../_base_/datasets/imagenet_sz224_4xbs64.py',
    '../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=10)

# interval for accumulate gradient
update_interval = 2  # total: 4 x bs256 x 2 accumulates = bs2048

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1.25e-3 * 2048 / 512,
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    },
)

# apex
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=2.5e-7 * 2048 / 512,
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=2.5e-7 / 1.25e-3,
    warmup_by_epoch=True,
    by_epoch=False)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
