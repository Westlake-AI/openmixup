_base_ = 'r50_CE_none_bs100.py'

# optimizer
optimizer = dict(
    type='Adan',
    lr=1e-3,
    weight_decay=0.02, eps=1e-8, betas=(0.98, 0.92, 0.99),
    max_grad_norm=5.0,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })
