# model settings
model = dict(
    type='SimMIM',
    backbone=dict(
        type='SimMIMSwinTransformer',
        arch='base',
        img_size=192,
        mask_layer=0, mask_token='learnable',
        drop_rate=0., drop_path_rate=0.,
        stage_cfgs=dict(block_cfgs=dict(window_size=6))),
    neck=dict(type='SimMIMNeck', in_channels=1024, encoder_stride=32),
    head=dict(type='SimMIMHead', encoder_in_channels=3))
