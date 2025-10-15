_base_ = [
    '../../../../_base_/datasets/imagenet/swin_sz224_4xbs256.py',
    '../../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MergeMix',
    pretrained=None,
    alpha=1.0,
    merge_num=49,
    mask_leaked=False,
    lam_scale=False,
    tome_in_mix=True,
    debug=False,
    backbone=dict(
        type='ToMeVisionTransformer',
        arch='deit-small',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=(11,),
        return_attn=True
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    head=dict(
        type='VisionTransformerClsHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        in_channels=384, num_classes=1000)
)

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=12)
# sampler = "RepeatAugSampler"  # this repo reproduce the performance without `repeated_aug`

# interval for accumulate gradient
update_interval = 1  # total: 4 x bs256 x 1 accumulates = bs1024

custom_hooks = [
    dict(type='SAVEHook',
        save_interval=1252 * 40,  # 20 ep
        iter_per_epoch=1252,
    ),
]


# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,  # lr = 5e-4 * (256 * 4) * 1 accumulate / 512 = 1e-3 / bs1024
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
    })

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# lr scheduler: Swim for DeiT
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,  # 1e-5 yields better performances than 1e-6
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
