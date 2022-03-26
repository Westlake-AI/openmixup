_base_ = '../../../_base_/datasets/imagenet/swin_sz224_4xbs256.py'

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(
        manifoldmix=dict(layer=(0, 3)),
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
        fmix=dict(decay_power=3, size=(224,224), max_soft=0., reformulate=False)
    ),
    backbone=dict(
        type='SwinTransformer',
        arch='tiny',
        img_size=224, drop_path_rate=0.2,
    ),
    head=dict(
        type='ClsMixupHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=True, in_channels=768, num_classes=1000)
)

# additional hooks
update_interval = 1  # interval for accumulate gradient
custom_hooks = [
    dict(type='EMAHook',  # EMA_W = (1 - m) * EMA_W + m * W
        momentum=0.99996,
        warmup='linear',
        warmup_iters=20 * 2503, warmup_ratio=0.9,  # warmup 20 epochs.
        update_interval=update_interval,
    ),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,  # lr = 5e-4 * (256 * 4) * 1 accumulate / 512 = 1e-3 / bs1024
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
    })
# apex
use_fp16 = True
# Notice: official ViT (DeiT or Swim) settings don't apply use_fp16=True. This repo use
#   use_fp16=True for fast training and better performances (around +0.1%).
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0),  # DeiT and Swim repos suggest max_norm=5.0
    update_interval=update_interval, use_fp16=use_fp16)

# lr scheduler: Swim for DeiT
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,  # 1e-5 yields better performances than 1e-6
    warmup='linear',
    warmup_iters=20, warmup_by_epoch=True,  # warmup 20 epochs.
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
