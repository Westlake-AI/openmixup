_base_ = '../../_base_/datasets/imagenet/mocov3_vit_sz224_bs64.py'

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

# interval for accumulate gradient
update_interval = 8  # total: 8 x bs64 x 8 accumulates = bs4096

# additional hooks
custom_hooks = [
    dict(type='CosineScheduleHook',  # update momentum
        end_momentum=1.0,
        adjust_scope=[0.05, 1.0],
        warming_up="constant",
        interval=update_interval)
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2.4e-3,  # bs4096
    betas=(0.9, 0.95), weight_decay=0.1,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.)
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
# optimizer args
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16, grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
