_base_ = './repmlp_base_8xb64_ep300.py'

model = dict(backbone=dict(deploy=True))
