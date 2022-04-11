_base_ = [
    '../_base_/models/vit_base_p16.py',
    '../_base_/datasets/imagenet_swin_sz224_8xbs128.py',
    '../_base_/default_runtime.py',
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3 * 1024 / 256,
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
    })

# learning policy
lr_config = dict(
    policy='StepFixCosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-4,
    warmup_by_epoch=True,
    by_epoch=False)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
