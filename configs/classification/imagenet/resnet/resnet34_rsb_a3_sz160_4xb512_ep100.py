_base_ = "resnet18_rsb_a3_sz160_4xb512_ep100.py"

# model settings
model = dict(backbone=dict(depth=34))
