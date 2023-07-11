_base_ = "../swin_t_mixups_sz224_bs100.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="resizemix",
    mix_args=dict(
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
    ),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
