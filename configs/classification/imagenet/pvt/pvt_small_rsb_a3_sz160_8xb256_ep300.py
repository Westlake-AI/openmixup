_base_ = [
    '../../_base_/models/pvt/pvt_small.py',
    '../../_base_/datasets/imagenet/rsb_a3_sz160_8xbs256.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    alpha=[0.1, 1.0,],
    mix_mode=["mixup", "cutmix",],
    head=dict(
        type='VisionTransformerClsHead',
        loss=dict(type='CrossEntropyLoss',  # mixup BCE loss (one-hot encoding)
            use_soft=False, use_sigmoid=True, loss_weight=1.0),
        multi_label=True, two_hot=False,
        in_channels=512, num_classes=1000),
)

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=10)

# additional hooks
update_interval = 1  # 256 x 8gpus x 1 accumulates = bs2048

# optimizer
optimizer = dict(
    type='LAMB', lr=0.008, weight_decay=0.02,
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
    })

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
