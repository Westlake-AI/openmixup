_base_ = '../repvgg_B1_4xb64_ep120.py'

model = dict(backbone=dict(deploy=True))
