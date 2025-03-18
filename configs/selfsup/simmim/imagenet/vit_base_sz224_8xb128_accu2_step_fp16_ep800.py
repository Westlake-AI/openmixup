_base_ = [
    '../../_base_/models/simmim/vit_base.py',
    '../../_base_/datasets/imagenet/simmim_sz224_p16_bs64.py',
    '../../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=128, workers_per_gpu=10)

# interval for accumulate gradient
update_interval = 2  # total: 8 x bs128 x 2 accumulates = bs2048

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
<<<<<<< HEAD
        save_interval=1252 * 10,  # plot every 10 ep
=======
        save_interval=1252 * 25,  # plot every 25 ep
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        iter_per_epoch=1252),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4 * 2048 / 512,  # 4e-4 for bs2048
    betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),
    })

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(
    update_interval=update_interval, grad_clip=dict(max_norm=5.0),
)

# lr scheduler
lr_config = dict(
    policy='step', step=[700,], gamma=0.1,
    warmup='linear',
    warmup_iters=10, warmup_by_epoch=True,
    warmup_ratio=5e-7 * 2048 / 512,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=800)
