_base_ = [
    '../_base_/models/r18_cifar.py',
    '../_base_/datasets/cifar10_sz32_bs100.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(head=dict(num_classes=10))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-4)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
