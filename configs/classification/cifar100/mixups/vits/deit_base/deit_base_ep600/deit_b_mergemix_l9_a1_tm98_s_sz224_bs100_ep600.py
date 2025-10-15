_base_ = [
    '../../../../../_base_/datasets/cifar100/sz224_randaug_bs100.py',
    '../../../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MergeMix',
    pretrained=None,
    alpha=1.0,
    merge_num=98,
    mask_leaked=False,
    lam_scale=True,
    tome_in_mix=False,
    swich_off=0.2,
    debug=False,
    backbone=dict(
        type='ToMeVisionTransformer',
        arch='deit-base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=(8, 11,),
        return_attn=True
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    head=dict(
        type='VisionTransformerClsHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=100, mode='original', loss_weight=1.0),
        in_channels=768, num_classes=100)
)

# interval for accumulate gradient
update_interval = 1

custom_hooks = [
    dict(type='SAVEHook',
        save_interval=500 * 20,  # 20 ep
        iter_per_epoch=500,
    ),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-4,
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

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=0.,
    warmup='linear',
    warmup_iters=20, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)