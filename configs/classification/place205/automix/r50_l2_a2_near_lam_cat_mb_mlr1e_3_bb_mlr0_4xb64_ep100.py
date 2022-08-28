_base_ = [
    '../../_base_/datasets/place205/basic_sz224_4xbs64.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='AutoMixup',
    pretrained=None,
    alpha=2.0,
    momentum=0.999,  # 0.999 to 0.99999
    mask_layer=2,
    mask_loss=0.1,  # using mask loss
    mask_adjust=0,
    lam_margin=0.08,  # degenerate to mixup when lam or 1-lam <= 0.08
    mask_up_override=None,  # If not none, override upsampling when train MixBlock
    debug=False,  # show attention and content map
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2,3),  # stage-3 for MixBlock, x-1: stage-x
        style='pytorch'),
    mix_block = dict(  # AutoMix
        type='PixelMixBlock',
        in_channels=1024, reduction=2, use_scale=True,
        unsampling_mode=['nearest',],  # str or list, train & test MixBlock
        lam_concat=True, lam_concat_v=False,  # AutoMix: lam cat q,k,v
        lam_mul=False, lam_residual=False, lam_mul_k=-1,  # SAMix lam: none
        value_neck_cfg=None,  # SAMix: non-linear value
        x_qk_concat=False, x_v_concat=False,  # SAMix x concat: none
        # att_norm_cfg=dict(type='BN'),  # norm after q,k (design for fp16, also conduct better performace in fp32)
        mask_loss_mode="L1", mask_loss_margin=0.1,  # L1 loss, 0.1
        frozen=False),
    head_one=dict(
        type='ClsHead',  # default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=2048, num_classes=205),
    head_mix=dict(  # backbone & mixblock
        type='ClsMixupHead',  # mixup, default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=2048, num_classes=205),
    head_weights=dict(
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        save_interval=9566 * 10,  # plot every 10 ep
        iter_per_epoch=9566,
    ),
    dict(type='CustomCosineAnnealingHook',  # 0.1 to 0
        attr_name="mask_loss", attr_base=0.1, by_epoch=False,  # by iter
        min_attr=0.,
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
                    'mix_block': dict(lr=0.1, momentum=0.9)}
)
# optimizer args
optimizer_config = dict(update_interval=1, grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# additional scheduler
addtional_scheduler = dict(
    policy='CosineAnnealing', min_lr=1e-3,
    paramwise_options=['mix_block'],
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
