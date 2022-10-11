# model settings
model = dict(
    type='BEiT',
    backbone=dict(
        type='BEiTViT',
        arch='base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        use_shared_rel_pos_bias=False,
        use_window=True, init_values=0.1,  # BEiT: use init_value and relative pos encoding
    ),
    neck=dict(type='BEiTNeck', in_channels=768, num_classes=8192),
    head=dict(
        type='BEiTHead',
        tokenizer_path='work_dirs/my_pretrains/beit_ckpt/dalle_encoder.pth',
    )
)
