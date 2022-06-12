# model settings
model = dict(
    type='SimMIM',
    backbone=dict(
        type='SimMIMViT',
        arch='base',
        detach=False,  # detach original x
        replace=False,  # use residual mask token
        mask_layer=0, mask_token='learnable',
        img_size=224,
        drop_rate=0., drop_path_rate=0.1,
        use_window=True, init_values=0.1,  # SimMIM: use init_value and relative pos encoding
    ),
    neck=dict(
        type='NonLinearMIMNeck',
        decoder_cfg=None,
        in_channels=768, in_chans=3, encoder_stride=16),
    head=dict(type='SimMIMHead', encoder_in_channels=3))
