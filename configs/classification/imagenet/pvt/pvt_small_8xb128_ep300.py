_base_ = './pvt_tiny_8xb128_ep300.py'

# model settings
model = dict(backbone=dict(arch='small'))
