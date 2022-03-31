# model settings
model = dict(
    type='MAE',
    backbone=dict(
        type='MAEViT',
        arch='base', patch_size=16, mask_ratio=0.75),
    neck=dict(
        type='MAEPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,  # 3/4 * eocoder depth
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(type='MAEPretrainHead', norm_pix=True, patch_size=16)
)
