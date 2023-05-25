_base_ = '../mobileone_s1_4xb64_ep300.py'

model = dict(backbone=dict(deploy=True))
