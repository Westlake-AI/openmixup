_base_ = [
    '../../_base_/models/deit3/deit3_large_p16_sz224.py',
    '../../_base_/datasets/imagenet/deit3_ft_sz224_8xbs128.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    backbone=dict(img_size=224),
    head=dict(
        type='VisionTransformerClsHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        in_channels=1024, num_classes=1000)
)

# data
data = dict(imgs_per_gpu=64, workers_per_gpu=6)

# additional hooks
update_interval = 1  # 64 x 8gpus x 1 accumulates = bs512

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-5 * 8,  # lr = 8e-5 / bs512
    weight_decay=0.1, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),
    })

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)
