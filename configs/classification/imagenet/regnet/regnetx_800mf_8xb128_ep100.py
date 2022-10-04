_base_ = './regnetx_400mf_8xb128_ep100.py'

# model settings
model = dict(
    backbone=dict(type='RegNet', arch='regnetx_800mf'),
    head=dict(in_channels=672, ))
