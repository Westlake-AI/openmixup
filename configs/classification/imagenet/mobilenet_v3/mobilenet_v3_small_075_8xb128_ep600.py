_base_ = [
    '../../_base_/models/mobilenet_v3/mobilenet_v3_small_075.py',
    '../../_base_/datasets/imagenet/mobilenet_v3_sz224_8xbs128.py',
    '../../_base_/default_runtime.py',
]

# data
data = dict(imgs_per_gpu=128, workers_per_gpu=10)

# optimizer
optimizer=dict(
    type='RMSprop',
    lr=0.064,
    alpha=0.9,
    momentum=0.9,
    eps=0.0316,
    weight_decay=1e-5)
optimizer_config = dict(grad_clip=None)

# lr scheduler
lr_config = dict(policy='step', by_epoch=True, gamma=0.973, step=2)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
