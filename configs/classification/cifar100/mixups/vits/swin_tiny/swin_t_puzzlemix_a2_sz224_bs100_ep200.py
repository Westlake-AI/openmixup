_base_ = "../swin_t_mixups_sz224_bs100.py"

# model settings
model = dict(
    alpha=2.0,
    mix_mode="puzzlemix",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
