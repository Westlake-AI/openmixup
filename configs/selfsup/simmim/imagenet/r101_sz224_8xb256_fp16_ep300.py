_base_ = [
    '../../_base_/models/simmim/r101.py',
    '../../_base_/datasets/imagenet/simmim_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# dataset
data = dict(
    imgs_per_gpu=256, workers_per_gpu=12,
)

# interval for accumulate gradient
update_interval = 1  # bs256 x 8gpus = bs2048

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        save_interval=626 * 25,  # plot every 25 ep
        iter_per_epoch=626),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4 * 2048 / 512,  # bs2048
    betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0.),
    })

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='StepFixCosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=10, warmup_by_epoch=True,
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
