_base_ = '../repvgg_B0_4xb64_ep120.py'

model = dict(backbone=dict(deploy=True))
