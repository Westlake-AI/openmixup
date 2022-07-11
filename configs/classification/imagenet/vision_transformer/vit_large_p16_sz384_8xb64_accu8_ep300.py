_base_ = [
    '../../_base_/models/vision_transformer/vit_large_p16_sz224.py',
    '../../_base_/datasets/imagenet/swin_sz384_8xbs64.py',
    '../../_base_/default_runtime.py',
]

# model
model = dict(backbone=dict(img_size=384))

# data
data = dict(imgs_per_gpu=64, workers_per_gpu=6)

# additional hooks
update_interval = 8  # 64 x 8gpus x 8 accumulates = bs4096

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.003,  # bs4096
    weight_decay=0.3,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=1.0), update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=0,
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=1e-4,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
