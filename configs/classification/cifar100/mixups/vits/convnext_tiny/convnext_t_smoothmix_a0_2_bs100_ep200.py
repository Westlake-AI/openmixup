_base_ = "../convnext_t_mixups_bs100.py"

# model settings
model = dict(
    alpha=0.2,
    mix_mode="smoothmix",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
