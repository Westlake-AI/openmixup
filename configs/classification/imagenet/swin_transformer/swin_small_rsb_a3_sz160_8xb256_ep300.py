_base_ = [
    # '../../_base_/models/swin_transformer/swin_small_sz224.py',
    '../../_base_/datasets/imagenet/rsb_a3_sz160_8xbs256.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.1, 1.0,],  # RSB A3
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(
        type='SwinTransformer',
        arch='small',
        img_size=160, stage_cfgs=dict(block_cfgs=dict(window_size=5)),
        drop_path_rate=0.3,
        out_indices=(3,),  # x-1: stage-x
        init_cfg=[
            dict(type='TruncNormal', layer=['Linear'], std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ],
    ),
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
    type='LAMB', lr=0.006, weight_decay=0.02,
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'absolute_pos_embed': dict(weight_decay=0.),
        'relative_position_bias_table': dict(weight_decay=0.),
    })

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

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
