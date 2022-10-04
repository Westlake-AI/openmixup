_base_ = [
    '../../_base_/models/repmlp/regmlp_base.py',
    '../../_base_/datasets/imagenet/basic_sz256_4xbs64.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(backbone=dict(img_size=256))

# data
data = dict(imgs_per_gpu=64, workers_per_gpu=6)

# additional hooks
update_interval = 1  # total: 8 x bs64 x 1 accumulates = bs512

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-4,  # 5e-4 / bs512
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })

# apex
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-6,
    warmup='linear',
    warmup_iters=20, warmup_by_epoch=True,
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
