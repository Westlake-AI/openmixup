_base_ = [
    '../_base_/models/r18.py',
    '../_base_/datasets/cifar100_sz224_4xbs64.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    backbone=dict(frozen_stages=4),
    head=dict(num_classes=100))

# optimizer
<<<<<<< HEAD
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.)
=======
optimizer = dict(type='SGD', lr=1.0, momentum=0.9, weight_decay=0.)
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
