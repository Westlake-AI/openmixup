_base_ = [
    '../../../_base_/datasets/imagenet/basic_sz224_4xbs64.py',
    '../../../_base_/default_runtime.py',
]

# models settings
model = dict(
    type='AdAutoMix',
    pretrained=None,
<<<<<<< HEAD
    alpha=2.0,
    mix_samples=2,   # mix samples number
    is_random=True,
=======
    alpha=1.0,
    mix_samples=2,
    is_random=False,  # mix samples number
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    momentum=0.999,  # 0.999 to 0.999999
    lam_margin=0.03,  # degenerate to mixup when
    mixup_radio=0.5,
    beta_radio=0.1,
    debug=True,
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
<<<<<<< HEAD
        out_indices=(2, 3),  # stage-3 for MixBlock, x-1: stage-x
=======
        out_indices=(2,3),  # stage-3 for MixBlock, x-1: stage-x
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        style='pytorch'),
    mix_block=dict(
        type='AdaptiveMask',
        in_channel=256,
        reduction=2,
        lam_concat=True,
<<<<<<< HEAD
        use_scale=True, unsampling_mode='bilinear',attn_norm_cfg=dict(type='BN'),
=======
        use_scale=True, unsampling_mode='bilinear',
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        scale_factor=16,  # 4 for r18 and rx50; 2 for wrn and 16 for vits
        frozen=False),
    head_one=dict(
        type='ClsHead',  # default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=1000),
    head_mix=dict(  # backbone & mixblock
        type='ClsMixupHead',  # mixup, default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=1000),
    head_weights=dict(
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        save_interval=50040,  # plot every 5004 x 10ep
        iter_per_epoch=5004,
    ),
<<<<<<< HEAD
    dict(type='CustomCosineAnnealingHook',  # 0.1 to 0
        attr_name="beta_radio", attr_base=0.1, by_epoch=False,  # by iter
        min_attr=0.,
    ),
=======
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    dict(type='CosineScheduleHook',
        end_momentum=0.99999,
        adjust_scope=[0.1, 1.0],
        warming_up="constant",
        interval=1)
]

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001,
                paramwise_options={
<<<<<<< HEAD
                    'mix_block': dict(lr=0.1, momentum=0.9)},)  # required parawise_option
# apex
use_fp16 = True
fp16=dict(type='mmcv', loss_scale='dynamic')
=======
                    'mix_block': dict(lr=0.1, momentum=0.9, weight_decay=0.0001)},)  # required parawise_option
# apex
use_fp16 = False
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
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
