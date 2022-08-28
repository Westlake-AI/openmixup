_base_ = "r50_l2_a2_bili_val_dp01_mb_mlr1e_4_bb_mlr0_4xb64_ep100.py"

# value_neck_cfg
conv1x1=dict(
    type="ConvNeck",
    in_channels=256, hid_channels=128, out_channels=1,  # MixBlock v
    num_layers=2, kernel_size=1,
    with_last_norm=False, norm_cfg=dict(type='BN'),  # default
    with_last_dropout=0.1, with_avg_pool=False, with_residual=False)  # no res + dropout

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
        depth=18,
        num_stages=4,
        out_indices=(2,3),  # stage-3 for MixBlock, x-1: stage-x
        style='pytorch'),
    mix_block = dict(  # SAMix
        type='PixelMixBlock',
        in_channels=256, reduction=2, use_scale=True,
        unsampling_mode=['bilinear',],  # str or list, tricks in SAMix
        lam_concat=False, lam_concat_v=False,  # AutoMix.V1: none
        lam_mul=True, lam_residual=True, lam_mul_k=-1,  # SAMix lam: mult + k=-1 (-1 for large datasets)
        value_neck_cfg=conv1x1,  # SAMix: non-linear value
        x_qk_concat=True, x_v_concat=False,  # SAMix x concat: q,k
        # att_norm_cfg=dict(type='BN'),  # norm after q,k (design for fp16, also conduct better performace in fp32)
        mask_loss_mode="L1+Variance", mask_loss_margin=0.1,  # L1+Var loss, tricks in SAMix
        frozen=False),
    head_one=dict(
        type='ClsHead',  # default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=5089),
    head_mix=dict(  # backbone
        type='ClsMixupHead',  # mixup, default CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=5089),
    head_mix_k=dict(  # mixblock
        type='ClsMixupHead',  # mixup, soft CE (onehot encoding)
        loss=dict(type='CrossEntropyLoss', use_soft=True, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=True,
        neg_weight=1,  # try neg (eta in SAMix)
        in_channels=512, num_classes=5089),
    head_weights=dict(
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# additional scheduler
addtional_scheduler = dict(
    policy='CosineAnnealing', min_lr=1e-3,
    paramwise_options=['mix_block'],
)
