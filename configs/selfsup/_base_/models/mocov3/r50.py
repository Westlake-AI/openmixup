# model settings
model = dict(
    type='MoCoV3',
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
        with_bias=False, with_last_bn=True, with_last_bn_affine=False,
        with_last_bias=False, with_avg_pool=True, vit_backbone=False),
    head=dict(
        type='MoCoV3Head',
        temperature=0.2,
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256, hid_channels=4096, out_channels=256,
            num_layers=2,
            with_bias=False, with_last_bn=True, with_last_bn_affine=False,
            with_last_bias=False, with_avg_pool=False))
)
