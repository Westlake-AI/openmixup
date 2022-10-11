# model settings
model = dict(
    type='SimMIM',
    backbone=dict(
        type='SimMIMSwinTransformer',
        arch='base',
        img_size=192,
        replace=False,  # use residual mask token
        mask_layer=0, mask_token='learnable',
        drop_rate=0., drop_path_rate=0.,
        stage_cfgs=dict(block_cfgs=dict(window_size=6))),
    neck=dict(
        type='NonLinearMIMNeck',
        decoder_cfg=None,
        in_channels=1024, in_chans=3, encoder_stride=32),
    head=dict(
        type='A2MIMHead',
        loss=dict(type='RegressionLoss', mode='l1_loss',
            loss_weight=1.0, reduction='none'),
        unmask_weight=0.,
        fft_weight=0.5,
        fft_focal=True,
        fft_unmask_weight=0.,  # unmask patches in the fft loss
        fft_unmask_replace='mixed',
        fft_reweight=False,
        encoder_in_channels=3,
    ))
