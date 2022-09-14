_base_ = './edgenext_small_sz256_8xb256_ep300.py'

# model settings
model = dict(
    backbone=dict(arch='xxsmall', drop_path_rate=0.1),
    head=dict(in_channels=168),
)
