_base_ = [
    '../../_base_/models/mocov3/r50.py',
    '../../_base_/datasets/imagenet/mocov3_cnn_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# dataset
data = dict(imgs_per_gpu=256, workers_per_gpu=8)

# interval for accumulate gradient
update_interval = 4  # total: 4 x bs256 x 4 accumulates = bs4096

# additional hooks
custom_hooks = [
    dict(type='CosineScheduleHook',  # update momentum
        end_momentum=1.0,
        adjust_scope=[0.05, 1.0],
        warming_up="constant",
        interval=update_interval)
]

# optimizer
optimizer = dict(
    type='LARS',
    lr=0.6 * 4096 / 256,  # bs4096
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

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=0.,
    warmup='linear',
    warmup_iters=10, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
