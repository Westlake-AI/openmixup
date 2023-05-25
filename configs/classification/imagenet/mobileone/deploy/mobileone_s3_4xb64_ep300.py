_base_ = '../mobileone_s3_4xb64_ep300.py'

model = dict(backbone=dict(deploy=True))
