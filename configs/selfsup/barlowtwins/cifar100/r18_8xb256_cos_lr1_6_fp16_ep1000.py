_base_ = [
    '../../_base_/models/barlowtwins/r18.py',
    '../../_base_/datasets/cifar100/byol_sz224_bs256.py',
    '../../_base_/default_runtime.py',
]

# interval for accumulate gradient
update_interval = 1  # total: 8 x bs256 x 1 accumulates = bs2048

# optimizer
optimizer = dict(
    type='LARS',
    lr=1.6,  # lr=1.6 for bs2048
    momentum=0.9, weight_decay=1e-6,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lr_mult=0.024, lars_exclude=True),
        'bias': dict(weight_decay=0., lr_mult=0.024, lars_exclude=True),
        # bn layer in ResNet block downsample module
        'downsample.1':
        dict(weight_decay=0, lr_mult=0.024, lars_exclude=True),
    })

# apex
use_fp16 = True
fp16 = dict(type='apex', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval, grad_clip=None)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=0.0016,
    warmup='linear',
    warmup_iters=10, warmup_by_epoch=True,
    warmup_ratio=1.6e-4,  # cannot be 0
)

# log, 50k / 4096
log_config = dict(interval=10)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1000)
