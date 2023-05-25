_base_ = [
    '../../_base_/models/efficientnet_v2/efficientnet_v2_b3.py',
    '../../_base_/datasets/imagenet/swin_sz224_8xbs128.py',
    '../../_base_/default_runtime.py',
]

# data
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandAugment_timm',
        input_size=240,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
]
test_pipeline = [
    dict(type='CenterCropForEfficientNet',
        size=300,
        efficientnet_style=True,
        interpolation='bicubic'),  # bicubic
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

# prefetch
prefetch = False

data = dict(
    train=dict(
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        pipeline=test_pipeline,
        prefetch=False,
    ))

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
runner = dict(type='EpochBasedRunner', max_epochs=300)
