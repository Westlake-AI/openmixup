_base_ = [
    '../../../_base_/datasets/imagenet/swin_sz224_8xbs128.py',
    '../../../_base_/default_runtime.py',
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
    switch_off=0.8,  # switch off mixblock (fixed)
    mask_up_override=None,
    debug=True,
    backbone=dict(
        type='SwinTransformer',
        arch='tiny',
        img_size=224, drop_path_rate=0.2,
        out_indices=(2,3),  # use stage-2 of 7x7x768
    ),
    mix_block = dict(  # AutoMix
        type='PixelMixBlock',
        in_channels=768, reduction=2, use_scale=True,
        unsampling_mode=['nearest',],  # str or list, train & test MixBlock, 'nearest' for AutoMix
        lam_concat=True, lam_concat_v=False,  # AutoMix.V1: lam cat q,k,v
        lam_mul=False, lam_residual=False, lam_mul_k=-1,  # SAMix lam: none
        value_neck_cfg=None,  # SAMix: non-linear value
        x_qk_concat=False, x_v_concat=False,  # SAMix x concat: none
        att_norm_cfg=None,  # Not use attention_norm for better performance
        mask_loss_mode="L1", mask_loss_margin=0.1,  # L1 loss, 0.1
        frozen=False),
    head_one=dict(
        type='ClsMixupHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=True,
        in_channels=768, num_classes=1000),
    head_mix=dict(
        type='ClsMixupHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=True,
        in_channels=768, num_classes=1000),
    head_weights=dict(
        decent_weight=[], accent_weight=[],
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
)

# dataset
data = dict(imgs_per_gpu=128, workers_per_gpu=10)

# interval for accumulate gradient
update_interval = 1  # total: 8 x bs128 x 1 accumulates = bs1024

custom_hooks = [
    dict(type='SAVEHook',
        save_interval=1252 * 20,  # 20 ep
        iter_per_epoch=1252,
    ),
    dict(type='CustomCosineAnnealingHook',  # 0.1 to 0
        attr_name="mask_loss", attr_base=0.1, min_attr=0., by_epoch=False,  # by iter
        update_interval=update_interval,
    ),
    dict(type='CosineScheduleHook',
        end_momentum=0.99999,  # 0.999 to 0.99999
        adjust_scope=[0.25, 1.0],
        warming_up="constant",
        update_interval=update_interval,
        interval=1)
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,  # lr = 5e-4 * (256 * 4) * 1 accumulate / 512 = 1e-3 / bs1024
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'absolute_pos_embed': dict(weight_decay=0.),
        'relative_position_bias_table': dict(weight_decay=0.),
        'mix_block': dict(lr=1e-3),
    })
# Sets `find_unused_parameters`: randomly switch off mixblock
find_unused_parameters = True

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# lr scheduler: Swim for DeiT
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=20, warmup_by_epoch=True,  # warmup 20 epochs.
    warmup_ratio=1e-5,
)

# additional scheduler
addtional_scheduler = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-4,  # 0.1 x lr
    paramwise_options=['mix_block'],
    warmup_iters=20, warmup_by_epoch=True,  # warmup 20 epochs
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
