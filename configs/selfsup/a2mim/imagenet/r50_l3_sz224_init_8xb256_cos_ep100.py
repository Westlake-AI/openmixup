_base_ = [
    '../../_base_/models/a2mim/r50.py',
    '../../_base_/datasets/imagenet/a2mim_rgb_m_sz224_rrc08_bs64.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    backbone=dict(
        mask_layer=3, mask_token="learnable",
        mask_init=1e-6,  # init residual gamma
))

# dataset
data = dict(
    imgs_per_gpu=256, workers_per_gpu=10,
    train=dict(
        feature_mode=None, feature_args=dict(),
        mask_pipeline=[
            dict(type='BlockwiseMaskGenerator',
                input_size=224, mask_patch_size=32, mask_ratio=0.6, model_patch_size=16,  # stage 3
                mask_color='mean', mask_only=False),
        ],
))

# interval for accumulate gradient
update_interval = 1

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        save_interval=626 * 10,  # plot every 10 ep
        iter_per_epoch=626),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=3e-4 * 2048 / 512,  # 1.2e-3 for bs2048
    betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0., lr_mult=1e-1,),
        'mask_gamma': dict(weight_decay=0., lr_mult=1e-1,),
    })

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='StepFixCosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=10, warmup_by_epoch=True,
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
