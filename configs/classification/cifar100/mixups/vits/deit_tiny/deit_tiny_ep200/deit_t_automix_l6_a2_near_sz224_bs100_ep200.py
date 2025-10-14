_base_ = [
    '../../../../../_base_/datasets/cifar100/sz224_randaug_bs100.py',
    '../../../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='AutoMixup',
    pretrained=None,
    alpha=2.0,
    momentum=0.999,
    mask_layer=2,  # dowmsampling to 1/16
    mask_loss=0.1,  # using loss
    mask_adjust=0,  # none for large datasets
    lam_margin=0.08,
    switch_off=1.0,  # switch off mixblock (fixed)
    mask_up_override=None,
    debug=True,
    backbone=dict(
        type='VisionTransformer',
        arch='deit-tiny',
        img_size=224, patch_size=16,
        drop_path_rate=0.1,
        out_indices=(5, 11),  # DeiT-S: 12 layers, use 6-layer for MixBlock
    ),
    mix_block = dict(  # AutoMix
        type='PixelMixBlock',
        in_channels=192, reduction=2, use_scale=True,
        unsampling_mode=['nearest',],  # str or list, train & test MixBlock, 'nearest' for AutoMix
        lam_concat=True, lam_concat_v=False,  # AutoMix.V1: lam cat q,k,v
        lam_mul=False, lam_residual=False, lam_mul_k=-1,  # SAMix lam: none
        x_qk_concat=False, x_v_concat=False,  # SAMix x concat: none
        att_norm_cfg=None,  # AutoMix: attention norm for fp16
        mask_loss_mode="L1", mask_loss_margin=0.1,  # L1 loss, 0.1
        frozen=False),
    head_one=dict(
        type='VisionTransformerClsHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        in_channels=192, num_classes=1000),
    head_mix=dict(
        type='VisionTransformerClsHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        in_channels=192, num_classes=1000),
    head_weights=dict(
        decent_weight=[], accent_weight=[],
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
)

# interval for accumulate gradient
update_interval = 1  # total: 8 x bs128 x 1 accumulates = bs1024

custom_hooks = [
    dict(type='SAVEHook',
        save_interval=500 * 20,  # 20 ep
        iter_per_epoch=500,
    ),
    dict(type='CustomCosineAnnealingHook',  # 0.1 to 0
        attr_name="mask_loss", attr_base=0.1, min_attr=0., by_epoch=False,  # by iter
        update_interval=update_interval,
    ),
    dict(type='CosineScheduleHook',
        end_momentum=0.99996,  # 0.999 to 0.99996
        adjust_scope=[0.25, 1.0],
        warming_up="constant",
        update_interval=update_interval,
        interval=1)
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'mix_block': dict(lr=0.05),
    })
# # Sets `find_unused_parameters`: randomly switch off mixblock
# find_unused_parameters = True

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# lr scheduler: Swim for DeiT
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# additional scheduler
addtional_scheduler = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=0.005,  # 0.1 x lr
    paramwise_options=['mix_block'],
    warmup_iters=5, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# validation hook
evaluation = dict(initial=False, save_best=None)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
