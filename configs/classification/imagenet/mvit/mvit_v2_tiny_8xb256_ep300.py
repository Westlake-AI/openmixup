_base_ = [
    '../../_base_/models/mvit/mvit_v2_tiny.py',
    '../../_base_/datasets/imagenet/swin_sz224_8xbs128.py',
    '../../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=10)

# additional hooks
update_interval = 1  # 256 x 8gpus x 1 accumulates = bs2048

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2.5e-4,  # lr = 0.00025 / bs2048
    weight_decay=0.03, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'rel_pos_h': dict(weight_decay=0.),
        'rel_pos_w': dict(weight_decay=0.),
    })

# apex
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=1.0), update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-6,
    warmup='linear',
    warmup_iters=70, warmup_by_epoch=True,  # warmup 70 epochs.
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
