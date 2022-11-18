_base_ = [
    '../_base_/models/r50_rsb_a2.py',
    '../_base_/datasets/imagenet_rsb_a2_ft_sz224_8xbs256.py',
    '../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=10)

# interval for accumulate gradient
update_interval = 1  # 256 x 8gpus x 1 accumulates = bs2048
custom_hooks = [
    dict(type='EMAHook',  # EMA_W = (1 - m) * EMA_W + m * W
        momentum=0.9999,
        warmup='linear',
        warmup_iters=20 * 626, warmup_ratio=0.9,  # warmup 20 epochs.
        update_interval=update_interval,
    ),
]

# optimizer
optimizer = dict(type='LAMB', lr=0.005, weight_decay=0.02,  # RSB A2
                 paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
                    'bias': dict(weight_decay=0.)})
# apex
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1.0e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
