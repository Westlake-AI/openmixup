_base_ = [
    '../../_base_/models/rotation_pred/r18.py',
    '../../_base_/datasets/imagenet/rotation-pred_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# interval for accumulate gradient
update_interval = 1  # total: 8 x bs64 x 1 accumulates = bs512

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.2,  # lr=0.2 / bs512
    weight_decay=1e-4, momentum=0.9)

<<<<<<< HEAD
# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale='dynamic')
=======
# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
# optimizer args
optimizer_config = dict(update_interval=update_interval, grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    step=[60, 80],
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5ep
    warmup_ratio=0.1,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
