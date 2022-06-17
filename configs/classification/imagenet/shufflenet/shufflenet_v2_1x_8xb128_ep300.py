_base_ = [
    '../../_base_/models/shufflenet/shufflenet_v2_1x.py',
    '../../_base_/datasets/imagenet/basic_sz224_4xbs64.py',
    '../../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=128, workers_per_gpu=8)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.5, momentum=0.9, weight_decay=0.00004,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })
optimizer_config = dict(grad_clip=None)

# lr scheduler
lr_config = dict(
    policy='poly',
    by_epoch=False, min_lr=0,
    warmup='constant',
    warmup_iters=5000,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
