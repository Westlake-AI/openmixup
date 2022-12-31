_base_ = "../deit_base_smooth_mix_8xb128.py"

# model settings
model = dict(
    alpha=0.2,
    mix_mode="fmix",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
