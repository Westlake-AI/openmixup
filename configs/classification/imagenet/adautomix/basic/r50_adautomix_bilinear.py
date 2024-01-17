_base_ = [
    '../../../_base_/datasets/imagenet/basic_sz224_4xbs64.py',
    '../../../_base_/default_runtime.py',
]

# models settings
model = dict(
    type='AdAutoMix',
    pretrained=None,
    alpha=1.0,
    mix_samples=2,
    is_random=False,   # mix samples number
    momentum=0.999,  # 0.999 to 0.999999
    lam_margin=0.03,  # degenerate to mixup when
    mixup_radio=0.5,
    beta_radio=0.1,
    debug=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),  # stage-3 for MixBlock, x-1: stage-x
        style='pytorch'),
    mix_block=dict(
        type='AdaptiveMask',
        in_channel=1024,
        reduction=2,
        lam_concat=True,
        use_scale=True, unsampling_mode='bilinear',
        att_norm_cfg=dict(type='BN'),
        scale_factor=16,  # 4 for r18 and rx50; 2 for wrn and 16 for vits
        frozen=False),
    head_one=dict(
        type='ClsHead',  # default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=2048, num_classes=1000),
    head_mix=dict(  # backbone & mixblock
        type='ClsMixupHead',  # mixup, default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=2048, num_classes=1000),
    head_weights=dict(
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        save_interval=50040,  # plot every 5004 x 10ep
        iter_per_epoch=5004,
    ),
    dict(type='CosineScheduleHook',
        end_momentum=0.99999,
        adjust_scope=[0.1, 1.0],
        warming_up="constant",
        interval=1)
]

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001,
                paramwise_options={
                    'mix_block': dict(lr=0.1, momentum=0.9)},)  # required parawise_option
# apex
use_fp16 = False
optimizer_config = dict(update_interval=1, grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# additional scheduler
addtional_scheduler = dict(
    policy='CosineAnnealing', min_lr=0.001,  # 0.1 x 1/100
    paramwise_options=['mix_block'],
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
