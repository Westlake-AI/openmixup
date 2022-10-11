# model settings
model = dict(
    type='MaskFeat',
    mim_target='HOG',  # HOG feature by SlowFast implementation with out_channels = 9 * 12
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
        in_channels=2048, in_chans=9 * 12, encoder_stride=32 // 16),  # hog
    head=dict(
        type='A2MIMHead',
        loss=dict(type='RegressionLoss', mode='mse_loss',
            loss_weight=1.0, reduction='none'),
        unmask_weight=0.,
        encoder_in_channels=9 * 12),  # hog
)
