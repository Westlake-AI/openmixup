_base_ = "resnet18_4xb64_step_ep100.py"

# model settings
model = dict(backbone=dict(depth=34))
