_base_ = '../mobileone_s2_4xb64_ep300.py'

model = dict(backbone=dict(deploy=True))
