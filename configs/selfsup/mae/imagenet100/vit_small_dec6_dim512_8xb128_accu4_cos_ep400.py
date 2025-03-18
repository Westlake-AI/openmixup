_base_ = [
    '../../_base_/models/mae/vit_small.py',
    '../../_base_/datasets/imagenet100/mae_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# dataset
data = dict(imgs_per_gpu=128, workers_per_gpu=8)

# interval for accumulate gradient
update_interval = 8  # total: 8 x bs128 x 4 accumulates = bs4096

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        save_interval=124 * 100,  # plot every 100 ep
        iter_per_epoch=124),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1.5e-4 * 4096 / 256,  # bs4096
    betas=(0.9, 0.95), weight_decay=0.05,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.)
    })

<<<<<<< HEAD
# apex
use_fp16 = False  # Notice that MAE get NAN loss when using fp16 on CIAFR-100
fp16 = dict(type='apex', loss_scale='dynamic')
=======
# fp16
use_fp16 = False  # Notice that MAE get NAN loss when using fp16 on CIAFR-100
fp16 = dict(type='mmcv', loss_scale='dynamic')
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
# optimizer args
optimizer_config = dict(update_interval=update_interval, grad_clip=None)

# lr scheduler
lr_config = dict(
    policy='StepFixCosineAnnealing',
    by_epoch=False, min_lr=0.,
    warmup='linear',
    warmup_iters=40, warmup_by_epoch=True,  # warmup 40ep when training 400 or more epochs
    warmup_ratio=1e-4,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)
