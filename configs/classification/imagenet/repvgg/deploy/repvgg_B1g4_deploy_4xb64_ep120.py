_base_ = '../repvgg_B1g4_4xb64_ep120.py'

model = dict(backbone=dict(deploy=True))
