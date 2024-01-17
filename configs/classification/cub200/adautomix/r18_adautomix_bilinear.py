_base_ = [
    '../../_base_/datasets/cub200/sz224_bs16.py',
    '../../_base_/default_runtime.py',
]

# models settings
model = dict(
    type='AdAutoMix',
    # pretrained='tools/resnet18-pytorch.pth',
    alpha=1.0,
    mix_samples=3,
    is_random=True,
    momentum=0.999,  # 0.999 to 0.999999
    lam_margin=0.03,  # degenerate to mixup when
    mixed_radio=0.5,
    beta_radio=0.3,
    debug=True,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(2,3),
        style='pytorch'),
    mix_block=dict(
        type='AdaptiveMask',
        in_channel=256,
        reduction=2,
        lam_concat=True,
        use_scale=True, unsampling_mode='bilinear',
        scale_factor=16,
        frozen=False),
    head_one=dict(
        type='ClsHead',  # default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=200),
    head_mix=dict(
        type='ClsMixupHead',
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=200),
    head_weights=dict(
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
         iter_per_epoch=500,
         save_interval=5000,  # plot every 500 x 10 ep
    ),
    dict(type='CosineScheduleHook',
        end_momentum=0.99999,
        adjust_scope=[0.1, 1.0],
        warming_up="constant",
        interval=1)
]

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005,
                paramwise_options={
                    'mix_block': dict(lr=0.1, momentum=0.9)},)  # required parawise_option
# apex
use_fp16 = False
optimizer_config = dict(update_interval=1, grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.0)

# additional scheduler
addtional_scheduler = dict(
    policy='CosineAnnealing', min_lr=0.001,  # 0.1 x 1/100
    paramwise_options=['mix_block'],
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)