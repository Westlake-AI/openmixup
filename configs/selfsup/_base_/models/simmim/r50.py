# model settings
model = dict(
    type='SimMIM',
    backbone=dict(
        type='MIMResNet',
        depth=50,
        mask_layer=0, mask_token='learnable',
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    neck=dict(type='SimMIMNeck', in_channels=2048, encoder_stride=32),
    head=dict(type='SimMIMHead', encoder_in_channels=3))
