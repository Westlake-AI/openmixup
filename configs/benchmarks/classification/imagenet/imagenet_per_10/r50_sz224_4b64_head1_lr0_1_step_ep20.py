_base_ = [
    '../../_base_/models/r50.py',
    '../../_base_/datasets/imagenet_sz224_4xbs64.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(backbone=dict(norm_cfg=dict(type='SyncBN')))

# dataset settings
data = dict(
    train=dict(
        data_source=dict(
            list_file='data/meta/ImageNet/train_labeled_10percent.txt',
        ))
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    paramwise_options={'\\Ahead.': dict(lr_mult=1)})

# learning policy
lr_config = dict(policy='step', step=[12, 16], gamma=0.2)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)
