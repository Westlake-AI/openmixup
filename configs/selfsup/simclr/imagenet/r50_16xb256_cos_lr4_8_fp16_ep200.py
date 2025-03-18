_base_ = [
    '../../_base_/models/simclr/r50.py',
    '../../_base_/datasets/imagenet/simclr_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# interval for accumulate gradient
update_interval = 1  # SimCLR cannot use grad accumulation, total: 16 x bs256 = bs4096

# optimizer
optimizer = dict(
    type='LARS',
    lr=0.3 * 16,  # lr=0.3 / bs256
    momentum=0.9, weight_decay=1e-6,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True),
    })

<<<<<<< HEAD
# apex
use_fp16 = True
fp16 = dict(type='apex', loss_scale='dynamic')
=======
# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
# optimizer args
optimizer_config = dict(update_interval=update_interval, grad_clip=None)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=0.,
    warmup='linear',
    warmup_iters=10, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
