_base_ = 'twins_pcpvt_base_8xb128_ep300.py'

# model settings
model = dict(
    backbone=dict(arch='large'),
    head=dict(in_channels=512),
)
