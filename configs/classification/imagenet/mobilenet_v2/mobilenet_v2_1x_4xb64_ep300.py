_base_ = [
    '../../_base_/models/mobilenet_v2/mobilenet_v2_1x.py',
    '../../_base_/datasets/imagenet/basic_sz224_4xbs64.py',
    '../../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=64, workers_per_gpu=6)

# optimizer
optimizer = dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004)
optimizer_config = dict(grad_clip=None)

# lr scheduler
lr_config = dict(policy='step', gamma=0.98, step=1)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
