_base_ = [
    '../../_base_/models/repvgg/repvgg_B3.py',
    '../../_base_/datasets/imagenet/autoaug_sz224_4xbs64.py',
    '../../_base_/default_runtime.py',
]

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=1, grad_clip=None)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=0.,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,
    warmup_ratio=0.25,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
