_base_ = './pvt_medium_8xb128_ep300.py'

# model settings
model = dict(backbone=dict(arch='large'))
