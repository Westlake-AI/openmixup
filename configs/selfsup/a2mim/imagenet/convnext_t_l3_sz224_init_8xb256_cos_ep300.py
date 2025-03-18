_base_ = [
    '../../_base_/models/a2mim/convnext_t.py',
    '../../_base_/datasets/imagenet/a2mim_rgb_m_sz224_rrc08_bs64.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    backbone=dict(
        mask_layer=3, mask_token="learnable",
        mask_init=1e-6,  # init residual gamma
    ),
    head=dict(
        fft_weight=0., fft_focal=False,
    ),
)

# dataset
data = dict(
    imgs_per_gpu=256, workers_per_gpu=10,
    train=dict(
        feature_mode=None, feature_args=dict(),
        mask_pipeline=[
            dict(type='BlockwiseMaskGenerator',
<<<<<<< HEAD
                input_size=224, mask_patch_size=32, mask_ratio=0.6, model_patch_size=32,  # stage 3 in MogaNet
=======
                input_size=224, mask_patch_size=32, mask_ratio=0.6, model_patch_size=32,  # stage 3
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
                mask_color='mean', mask_only=False),
        ],
))

# interval for accumulate gradient
update_interval = 1  # bs256 x 8gpus = bs2048

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
<<<<<<< HEAD
        save_interval=626 * 10,  # plot every 10 ep
=======
        save_interval=626 * 25,  # plot every 25 ep
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        iter_per_epoch=626),
]

# optimizer
optimizer = dict(
    type='AdamW',
<<<<<<< HEAD
    lr=1e-4 * 2048 / 512,  # 4e-3 for bs2048
=======
    lr=3e-4 * 2048 / 512,  # 3e-4 * 4 for bs2048
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0., lr_mult=1e-1,),
        'mask_gamma': dict(weight_decay=0., lr_mult=1e-1,),
    })

# fp16
use_fp16 = True
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
runner = dict(type='EpochBasedRunner', max_epochs=300)
