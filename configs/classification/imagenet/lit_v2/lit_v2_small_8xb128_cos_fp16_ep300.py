_base_ = [
    # '../../_base_/models/lit_v2/lit_v2_small.py',
    '../../_base_/datasets/imagenet/swin_sz224_4xbs256.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(
        type='LIT',
        arch='small',
        drop_path_rate=0.2,
        alpha=0.9,
        window_size=[0, 0, 2, 1],
        attention_types=[None, None, "HiLo", "HiLo"],
        init_values=1e-6,
    ),
    head=dict(
        type='ClsMixupHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=True,
        in_channels=768, num_classes=1000),
    # init_cfg=[
    #     dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
    #     dict(type='Constant', layer=['LayerNorm', 'BatchNorm'], val=1., bias=0.)
    # ],
)

# data
data = dict(imgs_per_gpu=128, workers_per_gpu=10)

# additional hooks
update_interval = 1  # 128 x 8gpus x 1 accumulates = bs1024

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,  # lr = 1e-3 / bs1024
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),
        'offset': dict(weight_decay=0., lr_mul=0.01),
    })

# apex
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=20, warmup_by_epoch=True,
    warmup_ratio=1e-6,
)

# validation hook
evaluation = dict(
    initial=True,
    interval=1,
    imgs_per_gpu=32,  # dconv im2col_step
    workers_per_gpu=4,
    eval_param=dict(topk=(1, 5)))

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
