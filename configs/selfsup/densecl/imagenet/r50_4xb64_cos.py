_base_ = [
    '../../_base_/models/densecl/r50.py',
    '../../_base_/datasets/imagenet/mocov2_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# interval for accumulate gradient
update_interval = 1  # total: 4 x bs64 x 1 accumulates = bs256

# additional hooks
custom_hooks = [
    dict(type='CustomFixedHook',
        attr_name="loss_lambda", attr_base=0.5,  # set to 0.5 after warmup
        warmup='constant', warmup_ratio=0, warmup_iters=1000,  # set to 0 for the first 1000 iters
        warmup_by_epoch=False,  # by iter
    ),
]

# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=1e-4, momentum=0.9)

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval, grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
