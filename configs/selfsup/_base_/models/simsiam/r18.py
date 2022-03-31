# model settings
model = dict(
    type='SimSiam',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    neck=dict(
        type='NonLinearNeck',
        in_channels=512, hid_channels=2048, out_channels=2048,
        num_layers=3,
        with_bias=True, with_last_bn=False, with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
                in_channels=2048, hid_channels=512, out_channels=2048,
                num_layers=2,
                with_avg_pool=False,
                with_bias=True, with_last_bn=False, with_last_bias=True))
)
