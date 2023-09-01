_base_ = [
    '../../_base_/models/simmim/vit_small.py',
    '../../_base_/datasets/imagenet/simmim_sz224_p16_bs64.py',
    '../../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=12)

# interval for accumulate gradient
update_interval = 1  # total: 8 x bs256 x 1 accumulates = bs2048

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        save_interval=626 * 25,
        iter_per_epoch=626),
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
use_fp16 = True
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
