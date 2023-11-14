_base_ = [
    '../../_base_/models/mocov3/vit_small.py',
    '../../_base_/datasets/stl10/mocov3_vit_sz96_bs256.py',
    '../../_base_/default_runtime.py',
]

# interval for accumulate gradient
update_interval = 2  # total: 8 x bs256 x 2 accumulates = bs4096

# additional hooks
custom_hooks = [
    dict(type='CosineScheduleHook',  # update momentum
        end_momentum=1.0,
        adjust_scope=[0.05, 1.0],
        warming_up="constant",
        update_interval=update_interval),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1.5e-4 * 4096 / 256,  # bs4096
    betas=(0.9, 0.95), weight_decay=0.1,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.)
    })

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(
    update_interval=update_interval, grad_clip=dict(max_norm=5.0),
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=0.,
    warmup='linear',
    warmup_iters=40, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# log, 50k / 4096
log_config = dict(interval=49)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1000)
