_base_ = "resnet50_rsb_a3_sz160_8xb256_ep100.py"

# model settings
model = dict(backbone=dict(depth=101))
