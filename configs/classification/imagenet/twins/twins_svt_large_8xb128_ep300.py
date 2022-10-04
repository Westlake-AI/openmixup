_base_ = 'twins_svt_base_8xb128_ep300.py'

# model settings
model = dict(
    backbone=dict(arch='large'),
    head=dict(in_channels=1024),
)
