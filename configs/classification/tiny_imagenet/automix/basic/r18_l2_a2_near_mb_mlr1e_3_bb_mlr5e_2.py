_base_ = [
    '../../../_base_/datasets/tiny_imagenet/sz64_bs100.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='AutoMixup',
    pretrained=None,
    alpha=2.0,
    momentum=0.999,  # 0.999 to 0.999999
    mask_layer=2,
    mask_loss=0.1,  # using mask loss
    mask_adjust=0,  # prob of adjusting bb mask in terms of lam by mixup, 0.25 for CIFAR
    lam_margin=0.08,  # degenerate to mixup when lam or 1-lam <= 0.08
    mask_up_override=None,  # If not none, override upsampling when train MixBlock
    debug=True,  # show attention and content map
    backbone=dict(
        type='ResNet_CIFAR',  # CIFAR version
        depth=18,
        num_stages=4,
        out_indices=(2,3),  # stage-3 for MixBlock, x-1: stage-x
        style='pytorch'),
    mix_block = dict(  # AutoMix
        type='PixelMixBlock',
        in_channels=256, reduction=2, use_scale=True,
        unsampling_mode=['nearest',],  # str or list
        lam_concat=False, lam_concat_v=False,  # AutoMix: no lam cat for small-scale datasets
        lam_mul=False, lam_residual=False, lam_mul_k=-1,  # SAMix lam: none
        value_neck_cfg=None,  # SAMix: non-linear value
        x_qk_concat=False, x_v_concat=False,  # SAMix x concat: none
        # att_norm_cfg=dict(type='BN'),  # norm after q,k (design for fp16, also conduct better performace in fp32)
        mask_loss_mode="L1", mask_loss_margin=0.1,  # L1 loss, 0.1
        frozen=False),
    head_one=dict(
        type='ClsHead',  # default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=200),
    head_mix=dict(  # backbone & mixblock
        type='ClsMixupHead',  # mixup, default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=200),
    head_weights=dict(
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# additional hooks
custom_hooks = [
    dict(type='CosineScheduleHook',
        end_momentum=0.999999,
        adjust_scope=[0.1, 1.0],
        warming_up="constant",
<<<<<<< HEAD
        interval=1),
=======
        update_interval=1),
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    dict(type='SAVEHook',
        iter_per_epoch=1000,
        save_interval=25000,  # plot every 500 x 25 ep
    )
]

# optimizer
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001,
            paramwise_options={'mix_block': dict(lr=0.1)})  # required parawise_option
# fp16
use_fp16 = False
# optimizer args
optimizer_config = dict(update_interval=1, grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing', min_lr=5e-2)  # adjust mlr for small-scale datasets

# additional scheduler
addtional_scheduler = dict(
    policy='CosineAnnealing', min_lr=1e-3,  # 0.1 x 1/100
    paramwise_options=['mix_block'],
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)
