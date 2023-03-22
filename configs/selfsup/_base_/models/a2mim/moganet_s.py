# model settings
model = dict(
    type='A2MIM',
    backbone=dict(
        type='MIMMogaNet',
        arch="small",
        init_value=1e-5,
        drop_path_rate=0.0,
        stem_norm_cfg=dict(type='BN', eps=1e-5),
        conv_norm_cfg=dict(type='BN', eps=1e-5),
        attn_force_fp32=True,  # force fp32 of gating for fp16 training
        replace=False,  # use residual mask token
        mask_layer=3, mask_token='learnable',
    ),
    neck=dict(
        type='NonLinearMIMNeck',
        decoder_cfg=None,
        kernel_size=1,
        in_channels=512, in_chans=3, encoder_stride=32),
    head=dict(
        type='A2MIMHead',
        loss=dict(type='RegressionLoss', mode='focal_l1_loss',
            loss_weight=1.0, reduction='none',
            activate='sigmoid', alpha=0.2, gamma=1.0, residual=False),
        unmask_weight=0.,
        fft_weight=0.5,
        fft_focal=True,
        fft_unmask_weight=1e-3,  # unmask patches in the fft loss
        fft_unmask_replace='mixed',
        fft_reweight=False,
        encoder_in_channels=3,
    ))
