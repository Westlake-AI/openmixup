# model settings
model = dict(
    type='CAE',
    base_momentum=0.,
    backbone=dict(
        type='CAEViT',
        arch='base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        qkv_bias=False,
        init_values=0.1,  # CAE: use init_value without qkv_bias
    ),
    neck=dict(
        type='CAENeck',
        patch_size=16,
        embed_dims=768,
        num_heads=12,
        regressor_depth=4,
        decoder_depth=4,
        mlp_ratio=4,
        init_values=0.1,
    ),
    head=dict(
        type='CAEHead',
        lambd=2,
        tokenizer_path='work_dirs/my_pretrains/beit_ckpt/dalle_encoder.pth',
    )
)
