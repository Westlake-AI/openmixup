_base_ = [
    '../../_base_/models/swav/r50.py',
    '../../_base_/datasets/imagenet/swav_mcrop-2-6_sz224_96_bs32.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='SwAV',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    neck=dict(
        type='SwAVNeck',
        in_channels=2048, hid_channels=2048, out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='SwAVHead',
        feat_dim=128,  # equal to neck['out_channels']
        epsilon=0.05,
        temperature=0.1,
        num_crops=[2, 6],)
)

# interval for accumulate gradient
update_interval = 8  # total: 8 x bs64 x 8 accumulates = bs4096

# additional hooks
custom_hooks = [
    dict(type='SwAVHook',
        priority='VERY_HIGH',
        batch_size=64,
        epoch_queue_starts=15,
        crops_for_assign=[0, 1],
        feat_dim=128,
        queue_length=3840)
]

# optimizer
optimizer = dict(
    type='LARS',
    lr=0.6 * 16,  # lr=0.6 / bs256
    momentum=0.9, weight_decay=1e-6,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True),
    })

<<<<<<< HEAD
# apex
use_fp16 = True
fp16 = dict(type='apex', loss_scale='dynamic')
=======
# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
# optimizer args
optimizer_config = dict(
    update_interval=update_interval, grad_clip=None,
    cancel_grad=dict(prototypes=2503),  # cancel grad of `prototypes` for 1 ep
)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=6e-4,
    warmup='linear',
    warmup_iters=10, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
