_base_ = 'r50_CE_none_bs100.py'

# optimizer
optimizer = dict(
    type='LAMB',
    lr=0.001,
    weight_decay=0.02,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })
