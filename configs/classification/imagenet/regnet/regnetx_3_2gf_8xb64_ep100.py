_base_ = [
    '../../_base_/models/regnet/regnetx_3_2gf.py',
    '../../_base_/datasets/imagenet/lighting_sz224_4xbs64.py',
    '../../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=64, workers_per_gpu=6)

# additional hooks
update_interval = 1  # 64 x 8gpus x 1 accumulates = bs512
custom_hooks = [
    dict(type='PreciseBNHook',
        num_samples=8192,
        update_all_stats=False,
        interval=1,
    ),
]

# optimizer
optimizer = dict(
    type='SGD', lr=0.4,
    momentum=0.9, weight_decay=5e-5, nesterov=True)

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=1, grad_clip=None)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=0.,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,
    warmup_ratio=0.1,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
