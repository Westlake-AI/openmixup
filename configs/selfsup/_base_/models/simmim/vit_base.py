# model settings
model = dict(
    type='SimMIM',
    backbone=dict(
        type='SimMIMViT',
        arch='base',
        replace=True,
        mask_layer=0, mask_token='learnable',
        img_size=224,
        drop_rate=0., drop_path_rate=0.1,
        use_window=True, init_values=0.1,  # SimMIM: use init_value and relative pos encoding
    ),
    neck=dict(type='SimMIMNeck', in_channels=768, encoder_stride=16),
    head=dict(type='SimMIMHead', encoder_in_channels=3))
