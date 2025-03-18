_base_ = [
    '../../../_base_/datasets/cifar100/sz32_bs100.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='AdAutoMix',
    pretrained=None,
    alpha=2.0,
    mix_samples=3,
    is_random=True,
    momentum=0.999,  # 0.999 to 0.999999
    lam_margin=0.03,  # degenerate to mixup when
    mixup_radio=0.5,
    beta_radio=0.3,
    debug=True,
    backbone=dict(
        type='WideResNet',
        first_stride=1,  # CIFAR version
        in_channels=3,
        depth=28, widen_factor=8,  # WRN-28-8, 128-256-512
        drop_rate=0.0,
        out_indices=(1,2),  # no conv-1, stage-2 for MixBlock, x-1: stage-x
    ),
    mix_block=dict(
        type='AdaptiveMask',
        in_channel=256,
        reduction=2,
        lam_concat=False,
        use_scale=True, unsampling_mode='bilinear',attn_norm_cfg=dict(type='BN'),

        scale_factor=2,
        frozen=False),
    head_one=dict(
        type='ClsHead',  # default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=100),
    head_mix=dict(  # backbone & mixblock
        type='ClsMixupHead',  # mixup, default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=100),
    head_weights=dict(
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# interval for accumulate gradient
update_interval = 1

# additional hooks
custom_hooks = [
    dict(type='CosineScheduleHook',
        end_momentum=0.999999,
        adjust_scope=[0.1, 1.0],
        warming_up="constant",
        update_interval=update_interval,
        interval=1),
    dict(type='SAVEHook',
        iter_per_epoch=500,
        save_interval=500*20,  # plot every 500 x 20 ep
    )
]

# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.001,  # lr=3e-2 + wd=1e-3 for WRN
                )

optimizer_config = dict(grad_clip=None, update_interval=update_interval,)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-3)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)
