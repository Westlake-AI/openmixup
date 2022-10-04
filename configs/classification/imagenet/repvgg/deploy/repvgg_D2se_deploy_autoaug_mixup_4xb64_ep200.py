_base_ = '../repvgg_D2se_autoaug_mixup_4xb64_ep200.py'

model = dict(backbone=dict(deploy=True))
