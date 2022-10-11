# model settings
model = dict(
    type='MaskFeat',
    mim_target='HOG',  # HOG feature by SlowFast implementation with out_channels = 9 * 12
    backbone=dict(
        type='SimMIMSwinTransformer',
        arch='base',
        img_size=224,
        mask_layer=0, mask_token='learnable',
        drop_rate=0., drop_path_rate=0.,
        stage_cfgs=dict(block_cfgs=dict(window_size=7))),
    neck=dict(
        type='NonLinearMIMNeck',
        decoder_cfg=None,
        in_channels=1024, in_chans=9 * 12, encoder_stride=32 // 16),  # hog
    head=dict(
        type='A2MIMHead',
        loss=dict(type='RegressionLoss', mode='mse_loss',
            loss_weight=1.0, reduction='none'),
        unmask_weight=0.,
        encoder_in_channels=9 * 12),  # hog
)
