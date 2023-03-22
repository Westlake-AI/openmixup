_base_ = [
    '../../_base_/models/resnet/resnet50_rsb_a3.py',
    '../../_base_/datasets/imagenet/rsb_a2_sz224_8xbs256.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    alpha=[0.2, 1.0,],
    mix_mode=["mixup", "cutmix",],
    backbone=dict(drop_path_rate=0.05),  # RSB A2
    head=dict(
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        ),
)

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=12)

# additional hooks
update_interval = 1  # 256 x 8gpus x 1 accumulates = bs2048

# optimizer
optimizer = dict(
    type='LAMB', lr=0.005, weight_decay=0.01,  # RSB A1
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1.0e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-5,
    by_epoch=True,  # timm decays by epoch
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
