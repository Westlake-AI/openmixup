_base_ = [
    '../../_base_/models/simclr/r18.py',
    '../../_base_/datasets/cifar10/simclr_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=8)

# interval for accumulate gradient
update_interval = 1  # SimCLR cannot use grad accumulation, total: 8 x bs256 = bs2048

# optimizer
optimizer = dict(
    type='LARS',
    lr=0.3 * 8,  # lr=0.3 / bs256
    momentum=0.9, weight_decay=1e-6,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True),
    })

# apex
use_fp16 = True
fp16 = dict(type='apex', loss_scale='dynamic')
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
runner = dict(type='EpochBasedRunner', max_epochs=1000)
