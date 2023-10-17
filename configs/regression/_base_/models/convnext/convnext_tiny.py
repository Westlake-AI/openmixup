# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='ConvNeXt',
        arch='tiny',
        out_indices=(3,),  # x-1: stage-x
        act_cfg=dict(type='GELU'),
        drop_path_rate=0.3,
        gap_before_final_norm=True,
    ),
    head=dict(
        type='RegHead',
        loss=dict(type='RegressionLoss', mode="l1_loss", loss_weight=1.0),
        with_avg_pool=False, in_channels=768, out_channels=1),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer=['LayerNorm', 'BatchNorm'], val=1., bias=0.)
    ],
)
