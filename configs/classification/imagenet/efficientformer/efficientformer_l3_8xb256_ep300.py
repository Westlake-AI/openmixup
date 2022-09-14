_base_ = './efficientformer_l1_8xb256_ep300.py'

# model settings
model = dict(
    backbone=dict(arch='l3'),
    head=dict(in_channels=512),
)
