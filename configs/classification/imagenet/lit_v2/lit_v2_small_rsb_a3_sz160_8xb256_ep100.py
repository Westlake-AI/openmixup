_base_ = [
    '../../_base_/models/lit_v2/lit_v2_small.py',
    '../../_base_/datasets/imagenet/rsb_a3_sz160_8xbs256.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    alpha=[0.1, 1.0,],  # RSB A3
    mix_mode=["mixup", "cutmix",],
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='CrossEntropyLoss',  # mixup BCE loss (one-hot encoding)
            use_soft=False, use_sigmoid=True, loss_weight=1.0),
        multi_label=True, two_hot=False,
        with_avg_pool=True,
        in_channels=768, num_classes=1000,
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        ]),
)

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=12)

# additional hooks
update_interval = 1  # 256 x 8gpus x 1 accumulates = bs2048

# optimizer
optimizer = dict(
    type='LAMB', lr=0.008, weight_decay=0.02,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),
        'offset': dict(lr_mul=0.1),
    })

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-6,
    by_epoch=False
)

# validation hook
evaluation = dict(
    initial=True,
    interval=1,
    imgs_per_gpu=25,  # dconv im2col_step
    workers_per_gpu=4,
    eval_param=dict(topk=(1, 5)))

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
