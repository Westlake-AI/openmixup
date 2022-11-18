_base_ = [
    '../_base_/models/vit_base_p16.py',
    '../_base_/datasets/imagenet_swin_ft_sz224_8xbs128.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(
    backbone=dict(  # for SimMIM & CAE
        mix_repeat=2,
        use_window=True, init_values=0.1,  # use relative pos encoding + init value
))

# data
data = dict(imgs_per_gpu=128, workers_per_gpu=8)

# interval for accumulate gradient
update_interval = 1  # total: 8 x bs128 x 1 accumulates = bs1024
custom_hooks = [
    dict(type='EMAHook',  # EMA_W = (1 - m) * EMA_W + m * W
        momentum=0.99996,
        warmup='linear',
        warmup_iters=20 * 1252, warmup_ratio=0.9,  # warmup 20 epochs.
        update_interval=update_interval,
    ),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2.5e-3 * 1024 / 256,  # 10e-4 for SimMIM
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),
    },
    constructor='TransformerFinetuneConstructor',
    model_type='vit',
    layer_decay=0.65)

# learning policy
lr_config = dict(
    policy='StepFixCosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=1e-6,
    warmup_by_epoch=True,
    by_epoch=False)

# apex
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(
    update_interval=update_interval, grad_clip=dict(max_norm=5.0),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
