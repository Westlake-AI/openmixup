_base_ = "swin_tiny_smooth_mix_4xb256_fp16.py"

# model settings
model = dict(
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
)
