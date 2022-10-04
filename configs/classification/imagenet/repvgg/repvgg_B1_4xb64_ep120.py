_base_ = './repvgg_B0_4xb64_ep120.py'

model = dict(
    backbone=dict(arch='B1'),
    head=dict(in_channels=2048),
)
