_base_ = [
    '../../_base_/models/a2mim/vit_small.py',
    '../../_base_/datasets/cifar100/a2mim_rgb_m_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    backbone=dict(
        mask_layer=0, mask_token='learnable'),
    head=dict(fft_weight=0.0)
)

# dataset
data = dict(
    imgs_per_gpu=256, workers_per_gpu=8,
    train=dict(
        feature_mode=None, feature_args=dict(),
        mask_pipeline=[
            dict(type='BlockwiseMaskGenerator',
                input_size=224, mask_patch_size=32, model_patch_size=16, mask_ratio=0.6,
                mask_color='mean', mask_only=False),
        ],
))

# interval for accumulate gradient
update_interval = 2  # total: 4 x bs256 x 2 accumulates = bs2048

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        save_interval=49 * 100,  # plot every 100 ep
        iter_per_epoch=49),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4 * 2048 / 512,  # 4e-4 for bs2048
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
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5 * 2048 / 512,
    warmup='linear',
    warmup_iters=10, warmup_by_epoch=True,  # warmup 10ep when training 100ep
    warmup_ratio=1e-6 * 2048 / 512,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1000)

# log, 50k / 4096
log_config = dict(interval=20)
