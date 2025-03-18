_base_ = [
    '../../_base_/models/resnet/resnet18.py',
<<<<<<< HEAD
    '../../_base_/datasets/imagenet/basic_sz224_bs256.py',
=======
    '../../_base_/datasets/imagenet/basic_sz224_4xbs64.py',
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    '../../_base_/default_runtime.py',
]

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=1, grad_clip=None)

# lr scheduler
lr_config = dict(policy='step', step=[30, 60, 90,])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
