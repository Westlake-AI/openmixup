_base_ = [
    '../../_base_/datasets/cifar100/sz32_bs100.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=1,
    mix_mode="vanilla",
    backbone=dict(
        type='WideResNet',  # normal
        first_stride=1,  # CIFAR version
        in_channels=3,
        depth=28, widen_factor=10,  # WRN-28-10, 160-320-640
        drop_rate=0.0,
        out_indices=(2,),  # no conv-1, x-1: stage-x
        frozen_stages=-1,
    ),
    head=dict(
        type='ClsHead',  # normal CE loss
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=640, num_classes=100)
)

# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.001)

# fp16
use_fp16 = True
optimizer_config = dict(update_interval=1, grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
