_base_ = "../swin_t_mixups_sz224_bs100.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="vanilla",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
