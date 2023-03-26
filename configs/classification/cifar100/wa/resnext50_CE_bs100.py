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
        type='ResNeXt_CIFAR',
        depth=50,
        groups=32, width_per_group=4,  # 32x4d
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    head=dict(
        type='ClsHead',  # normal CE loss
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=2048, num_classes=100)
)

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(update_interval=1, grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)

# Flops: 1.36 GFLOPs
# Params: 23.18 M
