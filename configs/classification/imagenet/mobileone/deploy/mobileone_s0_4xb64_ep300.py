_base_ = '../mobileone_s0_4xb64_ep300.py'

model = dict(backbone=dict(deploy=True))
