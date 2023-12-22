_base_ = [
    '../../../_base_/datasets/cifar100/sz32_bs100.py',
    '../../../_base_/default_runtime.py',
]

# models settings
model = dict(
    type='AdAutoMix',
    pretrained=None,
    alpha=1.0,
    mix_samples=3,   # mix samples number
    is_random=False,
    momentum=0.999,  # 0.999 to 0.999999
    lam_margin=0.03,
    mixup_radio=0.5,
    beta_radio=0.3,
    debug=True,
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(2,3),  # 2:[b,256,8,8]
        style='pytorch'),
    mix_block=dict(
        type='AdaptiveMask',
        in_channel=256,
        reduction=2,
        lam_concat=False,
        use_scale=True, unsampling_mode='bilinear',
        scale_factor=4,  # 4 for r18 and rx50; 2 for wrn and 16 for vits
        frozen=False),
    head_one=dict(
        type='ClsHead',  # default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=100),
    head_mix=dict(
        type='ClsMixupHead',
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=100),
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
        save_interval=500*25,  # plot every 500 x 25 ep
    )
]

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.05)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)
