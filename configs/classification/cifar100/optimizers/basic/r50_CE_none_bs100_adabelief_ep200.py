_base_ = 'r50_CE_none_bs100.py'

# optimizer
optimizer = dict(
    type='AdaBelief',
    lr=1e-3,
    weight_decay=0.0, eps=1e-8, betas=(0.9, 0.999),
    amsgrad=False, decoupled_decay=True, fixed_decay=False, rectify=True, degenerated_to_sgd=True)
