_base_ = 'r50_CE_none_bs100.py'

# optimizer
optimizer = dict(
    type='AdamP',
    lr=1e-3,
    betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    delta=0.1, wd_ratio=0.1, nesterov=False)
