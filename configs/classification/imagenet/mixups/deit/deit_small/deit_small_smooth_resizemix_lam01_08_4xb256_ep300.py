_base_ = "../deit_small_smooth_mix_4xb256_fp16.py"

# model settings
model = dict(
    alpha=1.0,
    mix_mode="resizemix",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
