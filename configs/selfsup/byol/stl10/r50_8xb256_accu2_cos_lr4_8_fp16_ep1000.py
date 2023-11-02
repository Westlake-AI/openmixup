_base_ = [
    '../../_base_/models/byol/r50.py',
    '../../_base_/datasets/stl10/byol_sz96_bs256.py',
    '../../_base_/default_runtime.py',
]

# dataset
data = dict(imgs_per_gpu=256, workers_per_gpu=8)

# interval for accumulate gradient
update_interval = 2  # total: 8 x bs256 x 2 accumulates = bs4096

# optimizer
optimizer = dict(
    type='LARS',
    lr=4.8,  # lr=4.8 / bs4096 for longer training
    momentum=0.9, weight_decay=1e-6,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True),
    })

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
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

# log, 50k / 4096
log_config = dict(interval=49)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1000)
