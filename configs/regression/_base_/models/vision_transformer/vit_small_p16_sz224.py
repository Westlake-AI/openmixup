# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='VisionTransformer',
        arch='base',
        img_size=224, patch_size=16,
        drop_path_rate=0.1,
    ),
    head=dict(
        type='RegHead',
        loss=dict(type='RegressionLoss', mode="l1_loss", loss_weight=1.0),
        with_avg_pool=False, in_channels=768, out_channels=1)
)
