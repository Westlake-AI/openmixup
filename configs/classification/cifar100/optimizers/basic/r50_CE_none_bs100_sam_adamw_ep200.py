_base_ = 'r50_CE_none_bs100.py'

data = dict(
    train=dict(data_source=dict(repeat=2)),  # repeat 2 times
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    rho=0.05, adaptive=False,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })

optimizer_config = dict(update_interval=2)  # repeat 2 times
