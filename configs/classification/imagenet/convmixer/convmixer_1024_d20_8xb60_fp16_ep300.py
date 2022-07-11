_base_ = [
    '../../_base_/models/convmixer/convmixer_1024_d20.py',
    '../../_base_/datasets/imagenet/convmixer_sz224_8xbs128.py',
    '../../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=80, workers_per_gpu=6)

# additional hooks
update_interval = 1  # 8 x 80gpus x 1 accumulates = bs640

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-2,  # lr = 1e-2 / bs640
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })

# apex
use_fp16 = True
fp16 = dict(type='apex', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=1.0), update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=20, warmup_by_epoch=True,  # warmup 20 epochs.
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
