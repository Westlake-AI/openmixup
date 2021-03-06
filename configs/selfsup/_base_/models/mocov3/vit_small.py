# model settings
model = dict(
    type='MoCoV3',
    base_momentum=0.99,
    backbone=dict(
        type='VisionTransformer',
        arch='mocov3-small',  # embed_dim = 384
        img_size=224,
        patch_size=16,
        stop_grad_conv1=True),
    neck=dict(
        type='NonLinearNeck',
        in_channels=384, hid_channels=4096, out_channels=256,
        num_layers=3,
        with_bias=False, with_last_bn=True, with_last_bn_affine=False,
        with_last_bias=False, with_avg_pool=False,
        vit_backbone=True),
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
