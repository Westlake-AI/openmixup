_base_ = "../deit_base_smooth_mix_8xb128.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="resizemix",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
