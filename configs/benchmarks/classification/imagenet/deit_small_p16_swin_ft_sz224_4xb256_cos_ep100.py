_base_ = [
    '../_base_/models/deit_small_p16.py',
    '../_base_/datasets/imagenet_swin_ft_sz224_8xbs128.py',
    '../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=10)  # total 4 x bs256 = bs1024, 4 GPU linear cls

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
    },
    constructor='TransformerFinetuneConstructor',
    model_type='vit',
    layer_decay=0.65)

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
