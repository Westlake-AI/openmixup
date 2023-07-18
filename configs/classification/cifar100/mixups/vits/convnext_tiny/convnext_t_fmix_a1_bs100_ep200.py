_base_ = "../convnext_t_mixups_bs100.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="fmix",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
