# model settings
model = dict(
    type='MaskFeat',
    mim_target='HOG',  # HOG feature by SlowFast implementation with out_channels = 9 * 12
    backbone=dict(
        type='SimMIMViT',
        arch='base',
        replace=True,
        mask_layer=0, mask_token='learnable',
        img_size=224,
        drop_rate=0., drop_path_rate=0.1,
        use_window=True, init_values=0.1,  # SimMIM: use init_value and relative pos encoding
    ),
    neck=dict(
        type='NonLinearMIMNeck',
        decoder_cfg=None,
        in_channels=768, in_chans=9 * 12, encoder_stride=16 // 16),  # HOG
    head=dict(
        type='A2MIMHead',
        loss=dict(type='RegressionLoss', mode='mse_loss',
            loss_weight=1.0, reduction='none'),
        unmask_weight=0.,
        encoder_in_channels=9 * 12),  # HOG
)
