_base_ = [
    'r50_sz224_4xb64_head1_lr0_1_step_ep20.py',
]

# optimizer
optimizer = dict(lr=0.01, paramwise_options={'\\Ahead.': dict(lr_mult=10)})
