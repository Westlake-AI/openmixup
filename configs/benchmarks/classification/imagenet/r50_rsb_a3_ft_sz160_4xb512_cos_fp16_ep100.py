_base_ = [
    '../_base_/models/r50_rsb_a3.py',
    '../_base_/datasets/imagenet_rsb_a3_ft_sz160_4xbs512.py',
    '../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=512, workers_per_gpu=10)

# interval for accumulate gradient
update_interval = 1  # total: 4 x bs512 x 1 accumulates = bs2048

# optimizer
optimizer = dict(type='LAMB', lr=0.008, weight_decay=0.02,
                 paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
                    'bias': dict(weight_decay=0.)})
# apex
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1.0e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
