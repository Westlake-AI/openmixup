_base_ = [
    '../../_base_/models/resnet/resnet50.py',
    '../../_base_/datasets/imagenet21k/basic_sz224_8xbs128.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    head=dict(
        type='ClsHead',  # CE + label smooth
        loss=dict(type='SemanticSoftmaxLoss',
            processor='imagenet21k',
            tree_path="work_dirs/my_pretrains/official/imagenet21k_miil_tree_winter21.pth",
            label_smooth_val=0.2, loss_weight=1.0),
        with_avg_pool=True,
        in_channels=2048, num_classes=10450,
    ),
)

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=12)

# additional hooks
update_interval = 1  # 256 x 8gpus x 1 accumulates = bs2048

# optimizer
optimizer = dict(
    type='LAMB', lr=0.006, weight_decay=0.02,
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
    min_lr=1e-5,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-5,
    by_epoch=True,  # timm decays by epoch
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=90)
