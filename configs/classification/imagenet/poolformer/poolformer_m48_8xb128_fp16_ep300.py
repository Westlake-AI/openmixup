_base_ = [
    '../../_base_/models/poolformer/poolformer_m48.py',
    '../../_base_/datasets/imagenet/poolformer_m_sz224_8xbs128.py',
    '../../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=128, workers_per_gpu=8)

# additional hooks
update_interval = 1  # total: 8 x bs128 x 1 accumulates = bs1024

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-3,  # 2 x 1e-3 / bs1024, (official )setting the learning rate as 2e-3 sometimes sees better performance
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'layer_scale': dict(weight_decay=0.),
    })

# apex
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
