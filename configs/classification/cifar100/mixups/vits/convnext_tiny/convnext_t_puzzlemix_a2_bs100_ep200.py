_base_ = "../convnext_t_mixups_bs100.py"

# model settings
model = dict(
    alpha=2.0,
    mix_mode="puzzlemix",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
