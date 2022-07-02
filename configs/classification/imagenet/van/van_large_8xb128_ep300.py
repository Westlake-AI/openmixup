_base_ = './van_small_8xb128_ep300.py'

# model settings
model = dict(
    backbone=dict(arch='large', drop_path_rate=0.2),
)
