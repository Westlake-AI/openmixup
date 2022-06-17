_base_ = [
    '../../_base_/models/vgg/vgg11.py',
    '../../_base_/datasets/imagenet/basic_sz224_4xbs64.py',
    '../../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=64, workers_per_gpu=6)

# additional hooks
update_interval = 1  # total: 4 x bs64 x 1 accumulates = bs256

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# lr scheduler
lr_config = dict(policy='step', step=[30, 60, 90,])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
