# model settings
model = dict(
    type='MaskFeat',
    mim_target='hog',
    backbone=dict(
        type='MIMResNet',
        depth=50,
        mask_layer=0, mask_token='learnable',
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    neck=dict(
        type='NonLinearMIMNeck',
        decoder_cfg=None,
        in_channels=2048, in_chans=9, encoder_stride=32 // 8),  # hog
    head=dict(
        type='MIMHead',
        loss=dict(type='RegressionLoss', mode='mse_loss',
            loss_weight=1.0, reduction='none'),
        unmask_weight=0.,
        encoder_in_channels=9),  # hog
)
