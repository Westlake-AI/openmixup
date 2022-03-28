_base_ = '../../_base_/datasets/imagenet/odc_sz224_bs64.py'

# model settings
model = dict(
    type='ODC',
    with_sobel=False,
    backbone=dict(
        type='ResNet_mmcls',
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    neck=dict(
        type='ODCNeck',
        in_channels=512, hid_channels=512, out_channels=256,
        with_avg_pool=True),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=False,  # already has avgpool in the neck
        in_channels=256, num_classes=10000),
    memory_bank=dict(
        type='ODCMemory',
        length=1281167, feat_dim=256, momentum=0.5,
        num_classes=10000, min_cluster=20, debug=False)
)

# interval for accumulate gradient
update_interval = 1  # total: 8 x bs64 x 1 accumulates = bs512

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.03 * 2,  # lr=0.03 / bs256
    weight_decay=1e-5, momentum=0.9,
    paramwise_options={
        '\\Ahead.': dict(momentum=0.),
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
# optimizer args
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16, grad_clip=None)

# learning policy
lr_config = dict(policy='step', step=[90], gamma=0.4)  # decay at 400ep for 440ep

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
