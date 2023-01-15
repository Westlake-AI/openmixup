_base_ = "../deit_small_smooth_mix_4xb256_fp16.py"

# model settings
model = dict(
    alpha=[0.8, 1.0],
    mix_mode=["mixup", "transmix"],  # using TransMix instead of CutMix
    mix_prob=[0.2, 0.8],
    backbone=dict(return_attn=True),  # return the attn map for TransMix
)

custom_hooks = [
    dict(type='EMAHook',  # EMA_W = (1 - m) * EMA_W + m * W
        momentum=0.99996,
        warmup='linear',
        warmup_iters=20 * 1252, warmup_ratio=0.9,
        update_interval=1,
    ),
]

# optimizer
optimizer = dict(
    weight_decay=0.05,
)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-4,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
