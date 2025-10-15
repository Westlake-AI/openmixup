_base_ = "../../deit_t_mixups_sz224_bs100.py"

# model settings
model = dict(
    alpha=0.2,
    mix_mode="smoothmix",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
