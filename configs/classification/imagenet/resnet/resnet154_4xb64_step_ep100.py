_base_ = "resnet50_4xb64_step_ep100.py"

# model settings
model = dict(backbone=dict(depth=154))
