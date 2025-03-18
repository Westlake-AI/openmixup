_base_ = "../deit_b_mixups_sz224_bs100.py"

# model settings
model = dict(
    alpha=[0.8, 1.0],
    mix_mode=["mixup", "transmix"],  # using TransMix instead of CutMix
    mix_prob=[0.2, 0.8],
    backbone=dict(return_attn=True),  # return the attn map for TransMix
)

# optimizer
optimizer = dict(
    weight_decay=0.05,
)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-4,
    warmup='linear',
    warmup_iters=20, warmup_by_epoch=True,  # warmup 20 epochs.
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
