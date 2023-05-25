_base_ = '../levit_128_p16_8xb256_ep300.py'

model = dict(backbone=dict(deploy=True), head=dict(deploy=True))
