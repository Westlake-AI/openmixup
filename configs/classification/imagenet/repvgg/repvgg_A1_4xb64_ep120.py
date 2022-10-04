_base_ = './repvgg_A0_4xb64_ep120.py'

model = dict(backbone=dict(arch='A1'))
