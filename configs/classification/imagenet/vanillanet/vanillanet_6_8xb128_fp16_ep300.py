_base_ = [
    '../../_base_/models/vanillanet/vanillanet_6.py',
    '../../_base_/datasets/imagenet/rsb_a2_sz224_8xbs256.py',
    '../../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=128, workers_per_gpu=10)

# additional hooks
update_interval = 1  # 128 x 8gpus x 1 accumulates = bs1024
custom_hooks = [
    dict(type='CustomFixStepCosineAnnealingHook',  # 1 to 0 (inverted)
        attr_name="cos_annealing",
        attr_base=1, min_attr=0, by_epoch=True, max_iters=100,  # decay 100 ep
    ),
    dict(type='EMAHook',  # EMA_W = (1 - m) * EMA_W + m * W
        momentum=0.99996,
        warmup_iters=5 * 1252, warmup_ratio=0.9,  # warmup 5 epochs.
        update_interval=update_interval,
    ),
]

# optimizer
optimizer = dict(
    type='LAMB',
    lr=0.0048, weight_decay=0.32,
    paramwise_options={
        '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.)})

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
