_base_ = [
    '../../../_base_/datasets/cifar10/sz32_bs100.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='AdAutoMix',
    pretrained=None,
    alpha=1.0,
    co_mix=3,  # mix samples number
    momentum=0.999,  # 0.999 to 0.999999
    mask_loss=0.0,  # using mask loss
    mask_adjust=0,
    lam_margin=0.03,  # degenerate to mixup when
    debug=True,
    backbone=dict(
        type='ResNeXt_CIFAR',
        depth=50,
        groups=32, width_per_group=4,  # 32x4d
        num_stages=4,
        out_indices=(2,3),  # stage-3 for MixBlock, x-1: stage-x
        style='pytorch'),
    mix_block=dict(
        type='AdaptiveMask',
        in_channel=1024,
        reduction=2,
        lam_mul=False, lam_mul_k=-1,
        lam_residual=False, lam_concat=False,
        use_scale=True, unsampling_mode='bilinear',
        scale_factor=4,
        mask_loss_mode="L1+Variance", mask_loss_margin=0.1,
        frozen=False),
    head_one=dict(
        type='ClsHead',  # default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=2048, num_classes=10),
    head_mix=dict(  # backbone & mixblock
        type='ClsMixupHead',  # mixup, default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=2048, num_classes=10),
    head_weights=dict(
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# additional hooks
custom_hooks = [
    dict(type='CosineScheduleHook',
        end_momentum=0.999999,
        adjust_scope=[0.1, 1.0],
        warming_up="constant",
        interval=1),
    dict(type='SAVEHook',
        iter_per_epoch=500,
        save_interval=12500,  # plot every 500 x 25 ep
    )
]

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.05)  # min_lr=5e-2 for the momentum encoder

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=800)
