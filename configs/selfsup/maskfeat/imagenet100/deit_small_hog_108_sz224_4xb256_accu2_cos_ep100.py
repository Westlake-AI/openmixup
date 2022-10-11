_base_ = [
    '../../_base_/models/maskfeat/deit_small.py',
    '../../_base_/datasets/imagenet100/mim_feat_sz224_p16_bs64.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    mim_target='HOG',  # HOG feature by SlowFast implementation with out_channels = 9 * 12
    neck=dict(
        type='NonLinearMIMNeck',
        decoder_cfg=None,
        in_channels=384, in_chans=9 * 12, encoder_stride=16 // 16),  # HOG
    head=dict(
        type='A2MIMHead',
        loss=dict(type='RegressionLoss', mode='mse_loss',
            loss_weight=1.0, reduction='none'),
        unmask_weight=0.,
        encoder_in_channels=9 * 12),  # HOG
)

# dataset
data = dict(
    imgs_per_gpu=256, workers_per_gpu=10,
    train=dict(feature_mode=None, feature_args=dict()),
)

# interval for accumulate gradient
update_interval = 2  # total: 4 x bs256 x 2 accumulates = bs2048

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4 * 2048 / 256,  # 1.6e-3 for bs2048
    betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),
    })

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(
    update_interval=update_interval, grad_clip=dict(max_norm=0.02),
)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=30, warmup_by_epoch=True,
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=800)
