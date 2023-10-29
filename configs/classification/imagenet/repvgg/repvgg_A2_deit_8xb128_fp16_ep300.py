_base_ = [
    '../../_base_/models/repvgg/repvgg_A0.py',
    '../../_base_/datasets/imagenet/deit_adan_sz224_8xbs256.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    alpha=[0.1, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(arch='A2'),
    head=dict(
        type='ClsHead',
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=True, multi_label=True, two_hot=False,
        in_channels=1408, num_classes=1000),
)

# data
data = dict(imgs_per_gpu=128, workers_per_gpu=10)

# additional hooks
update_interval = 1  # 128 x 8gpus x 1 accumulates = bs1024

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval, grad_clip=None)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
