_base_ = './repvgg_B3_autoaug_mixup_4xb64_ep200.py'

model = dict(
    backbone=dict(arch='B2g4'),
    head=dict(in_channels=2560),
)
