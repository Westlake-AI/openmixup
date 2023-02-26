_base_ = [
    '../../_base_/datasets/imagenet/rsb_a2_sz224_8xbs256.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.1, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_prob=None,  # None for random applying
    mix_args=dict(),
    backbone=dict(
        type='ResNet',  # normal
        depth=50,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN'),
        drop_path_rate=0.05,  # RSB A2
        style='pytorch'),
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='CrossEntropyLoss',  # mixup BCE loss (one-hot encoding)
            use_soft=False, use_sigmoid=True, loss_weight=1.0),
        with_avg_pool=True, multi_label=True, two_hot=False,
        in_channels=2048, num_classes=1000)
)

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=12)

# additional hooks
update_interval = 1  # 256 x 8gpus x 1 accumulates = bs2048

# optimizer
optimizer = dict(
    type='LAMB', lr=0.005, weight_decay=0.02,  # RSB A2
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
    min_lr=0.,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-5,
    by_epoch=False
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
