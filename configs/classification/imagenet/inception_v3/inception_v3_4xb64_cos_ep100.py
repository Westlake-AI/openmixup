_base_ = [
    '../../_base_/datasets/imagenet/basic_sz299_4xbs64.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='InceptionV3',
        num_classes=1000,
        aux_logits=False),
    head=dict(
        type='ClsHead',  # normal CE loss
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=False, multi_label=False, in_channels=None, num_classes=None)
)

# data
data = dict(imgs_per_gpu=64, workers_per_gpu=4)

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=1, grad_clip=None)

# lr scheduler
lr_config = dict(policy='CosineAnnealing', min_lr=1e-6)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
