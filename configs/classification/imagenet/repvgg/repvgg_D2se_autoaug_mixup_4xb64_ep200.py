_base_ = './repvgg_B3_autoaug_mixup_4xb64_ep200.py'

model = dict(backbone=dict(arch='D2se'))
