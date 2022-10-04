_base_ = "pvt_tiny_smooth_mix_4xb256.py"

# model settings
model = dict(backbone=dict(arch='small'))
