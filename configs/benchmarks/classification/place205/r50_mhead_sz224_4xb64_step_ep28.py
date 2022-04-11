_base_ = [
    '../_base_/models/r50_multihead.py',
    '../_base_/datasets/place205_sz224_4xbs64.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    backbone=dict(frozen_stages=4),
    head=dict(
        norm_cfg=dict(type='SyncBN', momentum=0.1, affine=False),
        num_classes=205))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9, weight_decay=1e-4, nesterov=True,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
    },
)

# learning policy
lr_config = dict(policy='step', step=[7, 14, 21])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=28)
