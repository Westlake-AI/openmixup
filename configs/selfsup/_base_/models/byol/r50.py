# model settings
model = dict(
    type='BYOL',
    base_momentum=0.99,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048, hid_channels=4096, out_channels=256,
        num_layers=2,
        with_bias=True, with_last_bn=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
                in_channels=256, hid_channels=4096, out_channels=256,
                num_layers=2,
                with_bias=True, with_last_bn=False, with_avg_pool=False))
)
