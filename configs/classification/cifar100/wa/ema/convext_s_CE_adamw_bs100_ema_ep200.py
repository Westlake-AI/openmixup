_base_ = '../convext_s_CE_adamw_bs100.py'

# additional hooks
custom_hooks = [
    dict(type='EMAHook',  # EMA_W = (1 - m) * EMA_W + m * W
        momentum=0.9999,
        warmup='linear',
        warmup_iters=5 * 500, warmup_ratio=0.9,  # warmup 5 epochs.
        update_interval=1,  # bs100 x 1gpu
    ),
]

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
