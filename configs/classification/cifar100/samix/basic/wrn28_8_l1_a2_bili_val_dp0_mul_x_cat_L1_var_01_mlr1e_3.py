_base_ = [
    '../../../_base_/datasets/cifar100/sz32_bs100.py',
    '../../../_base_/default_runtime.py',
]

# value_neck_cfg
conv1x1=dict(
    type="ConvNeck",
    in_channels=256, hid_channels=128, out_channels=1,  # MixBlock v
    num_layers=2, kernel_size=1,
    with_last_norm=False, norm_cfg=dict(type='BN'),  # default
    with_last_dropout=0, with_avg_pool=False, with_residual=False)  # no res & dropout

# model settings
model = dict(
    type='AutoMixup',
    pretrained=None,
    alpha=2.0,
    momentum=0.999,  # 0.999 to 0.999999
    mask_layer=1,  # for WRN
    mask_loss=0.1,  # using mask loss
    mask_adjust=0,  # prob of adjusting bb mask in terms of lam by mixup, 0.25 for CIFAR
    lam_margin=0.08,  # degenerate to mixup when lam or 1-lam <= 0.08
    mask_up_override=None,  # If not none, override upsampling when train MixBlock
    debug=True,  # show attention and content map
    backbone=dict(
        type='WideResNet',
        first_stride=1,  # CIFAR version
        in_channels=3,
        depth=28, widen_factor=8,  # WRN-28-8, 128-256-512
        drop_rate=0.0,
        out_indices=(1,2,),  # no conv-1, stage-2 for Mixer, x-1: stage-x
    ),
    mix_block = dict(  # SAMix
        type='PixelMixBlock',
        in_channels=256, reduction=2, use_scale=True,
        unsampling_mode=['bilinear',],  # str or list, tricks in SAMix
        lam_concat=False, lam_concat_v=False,  # AutoMix.V1: none
        lam_mul=True, lam_residual=True, lam_mul_k=0.25,  # SAMix lam: mult + k (0.25 for CIFAR)
        value_neck_cfg=conv1x1,  # SAMix: non-linear value
        x_qk_concat=True, x_v_concat=False,  # SAMix x concat: q,k
        # att_norm_cfg=dict(type='BN'),  # norm after q,k (design for fp16, also conduct better performace in fp32)
        mask_loss_mode="L1+Variance", mask_loss_margin=0.1,  # L1+Var loss, tricks in SAMix
        frozen=False),
    head_one=dict(
        type='ClsHead',  # default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=100),
    head_mix=dict(  # backbone
        type='ClsMixupHead',  # mixup, default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=100),
    head_mix_k=dict(  # mixblock
        type='ClsMixupHead',  # mixup, soft CE (onehot encoding)
        loss=dict(type='CrossEntropyLoss', use_soft=True, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=True,
        neg_weight=0.5,  # try neg (eta in SAMix)
        in_channels=512, num_classes=100),
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
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.001,  # lr=3e-2 + wd=1e-3 for WRN
                paramwise_options={
                    'mix_block': dict(lr=0.03, momentum=0)})  # momentum=0 performs better in 400ep
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-3)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)
