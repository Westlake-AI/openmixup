_base_ = [
    '../../_base_/models/odc/r50.py',
    '../../_base_/datasets/imagenet/odc_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# interval for accumulate gradient
update_interval = 1  # total: 8 x bs64 x 1 accumulates = bs512

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.03 * 2,  # lr=0.03 / bs256
    weight_decay=1e-5, momentum=0.9,
    paramwise_options={
        '\\Ahead.': dict(momentum=0.),
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval, grad_clip=None)

# learning policy
lr_config = dict(policy='step', step=[400], gamma=0.4)  # decay at 400ep for 440ep

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=440)
