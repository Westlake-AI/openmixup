_base_ = 'r50_CE_none_bs100.py'

# optimizer
optimizer = dict(
    type='Adafactor',
    lr=1e-3,
    eps=1e-30, eps_scale=1e-3, clip_threshold=1.0,
    decay_rate=-0.8, betas=None, weight_decay=0.0, scale_parameter=True, warmup_init=False)
