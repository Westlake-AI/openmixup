_base_ = 'r50_CE_none_bs100.py'

data = dict(
    train=dict(data_source=dict(repeat=2)),  # repeat 2 times
)

# optimizer
optimizer = dict(
    type='SAMSGD', lr=0.1, momentum=0.9, weight_decay=0.0001,
    rho=0.05, adaptive=False)

optimizer_config = dict(update_interval=2)  # repeat 2 times
