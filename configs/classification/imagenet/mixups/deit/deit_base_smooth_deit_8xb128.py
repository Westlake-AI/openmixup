_base_ = "deit_base_smooth_mix_8xb128.py"

# model settings
model = dict(
    alpha=[0.8, 1.0,],  # deit setting
    mix_mode=["mixup", "cutmix",],
)
