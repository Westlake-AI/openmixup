_base_ = "../deit_b_mixups_sz224_bs100.py"

# model settings
model = dict(
    alpha=2.0,
    mix_mode="puzzlemix",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
