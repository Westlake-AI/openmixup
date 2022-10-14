_base_ = [
    '../../_base_/datasets/imagenet/basic_sz224_4xbs64.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='ResNet',  # normal
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style='pytorch'),
    head=dict(
        type='ClsHead',  # normal CE loss
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=1000)
)

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=1, grad_clip=None)

# lr scheduler
lr_config = dict(policy='CosineAnnealing', min_lr=0)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
