_base_ = '../../_base_/datasets/imagenet/relative-loc_sz224_bs64.py'

# model settings
model = dict(
    type='RelativeLoc',
    backbone=dict(
        type='ResNet_mmcls',
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='BN'),
        style='pytorch'),
    neck=dict(
        type='RelativeLocNeck',
        in_channels=512, out_channels=4096,
        with_avg_pool=True),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=False,  # already has avgpool in the neck
        in_channels=4096, num_classes=8),
)

# interval for accumulate gradient
update_interval = 1  # total: 8 x bs64 x 1 accumulates = bs512

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.2,  # lr=0.2 / bs512
    weight_decay=1e-4, momentum=0.9,
    paramwise_options={
        '\\Aneck.': dict(weight_decay=5e-4),
        '\\Ahead.': dict(weight_decay=5e-4),
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
# optimizer args
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16, grad_clip=None)

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
