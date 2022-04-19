_base_ = [
    '../../_base_/models/maskfeat/swin_tiny.py',
    '../../_base_/datasets/imagenet100/mim_feat_sz192_bs64.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    mim_target='sobel',
    neck=dict(
        type='NonLinearMIMNeck',
        decoder_cfg=None,
        in_channels=768, in_chans=2, encoder_stride=32),  # sobel
    head=dict(
        type='MIMHead',
        loss=dict(type='RegressionLoss', mode='mse_loss', loss_weight=1.0, reduction='none'),
        encoder_in_channels=2),  # sobel
)

# dataset
data = dict(
    imgs_per_gpu=64, workers_per_gpu=4,
    train=dict(feature_mode=None, feature_args=dict()),
)

# interval for accumulate gradient
update_interval = 4  # total: 8 x bs64 x 4 accumulates = bs2048

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        save_interval=495 * 20,  # plot every 20ep
        iter_per_epoch=495),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4 * 2048 / 512,  # bs2048
    betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0.),
        'absolute_pos_embed': dict(weight_decay=0.),
        'relative_position_bias_table': dict(weight_decay=0.0)
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
# optimizer args
optimizer_config = dict(
    update_interval=update_interval, use_fp16=use_fp16,
    grad_clip=dict(max_norm=5.0),
)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5 * 2048 / 512,
    warmup='linear',
    warmup_iters=10, warmup_by_epoch=True,  # warmup 10ep when training 100ep
    warmup_ratio=1e-6 / 2e-4,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
