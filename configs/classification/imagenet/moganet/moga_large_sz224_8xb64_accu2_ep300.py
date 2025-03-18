_base_ = [
    '../../_base_/models/moganet/moga_large.py',
    '../../_base_/datasets/imagenet/moga_sz224_8xbs128.py',
    '../../_base_/default_runtime.py',
]
<<<<<<< HEAD

=======
model = dict(backbone=dict(attn_force_fp32=False))  # force fp32 of gating for fp16 training
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
# data
data = dict(imgs_per_gpu=64, workers_per_gpu=8)

# additional hooks
update_interval = 2  # 64 x 8gpus x 2 accumulates = bs1024
custom_hooks = [
    dict(type='PreciseBNHook',
        num_samples=8192,
        update_all_stats=False,
        interval=1,
    ),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,  # lr = 1e-3 / bs1024
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'layer_scale': dict(weight_decay=0.),
        'scale': dict(weight_decay=0.),
    })

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
