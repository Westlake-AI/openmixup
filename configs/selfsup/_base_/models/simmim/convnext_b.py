# model settings
model = dict(
    type='SimMIM',
    backbone=dict(
        type='MIMConvNeXt',
        arch="base",
        out_indices=(3,),  # x-1: stage-x
        act_cfg=dict(type='GELU'),
        drop_path_rate=0.0,
        gap_before_final_norm=False,
        replace=False,  # use residual mask token
        mask_layer=0, mask_token='learnable',
    ),
    neck=dict(type='SimMIMNeck', in_channels=1024, encoder_stride=32),
    head=dict(type='SimMIMHead', encoder_in_channels=3))
