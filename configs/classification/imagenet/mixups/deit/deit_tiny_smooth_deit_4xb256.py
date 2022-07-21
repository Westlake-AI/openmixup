_base_ = "deit_tiny_smooth_mix_4xb256.py"

# model settings
model = dict(
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
)
