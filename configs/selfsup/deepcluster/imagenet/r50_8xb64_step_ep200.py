_base_ = '../../_base_/datasets/imagenet/deepcluster_sz224_bs64.py'

# model settings
model = dict(
    type='DeepCluster',
    backbone=dict(
        type='ResNet_mmcls',
        depth=50,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='BN'),
        style='pytorch'),
    neck=dict(type='AvgPoolNeck'),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=False,  # already has avgpool in the neck
        in_channels=2048, num_classes=10000)
)

# interval for accumulate gradient
update_interval = 1

# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=1e-4, momentum=0.9)

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
# optimizer args
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16, grad_clip=None)

# learning policy
lr_config = dict(policy='step', step=[120, 160])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
