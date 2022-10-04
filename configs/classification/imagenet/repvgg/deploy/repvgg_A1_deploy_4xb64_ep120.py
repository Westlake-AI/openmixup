_base_ = '../repvgg_A1_4xb64_ep120.py'

model = dict(backbone=dict(deploy=True))
