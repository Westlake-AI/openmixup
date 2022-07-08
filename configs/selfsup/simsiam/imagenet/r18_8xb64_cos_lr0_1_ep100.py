_base_ = [
    '../../_base_/models/simsiam/r18.py',
    '../../_base_/datasets/imagenet/mocov2_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# interval for accumulate gradient
update_interval = 1  # total: 8 x bs64 x 1 accumulates = bs512

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.05 * 2,  # lr=0.05 / bs256
    weight_decay=1e-4, momentum=0.9,
    paramwise_options={
        'predictor': dict(lr=0.05 * 2,),  # fix preditor lr
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval, grad_clip=None)

# additional lr scheduler (parawise_options required in optimizer)
addtional_scheduler = dict(
    policy='Fixed', paramwise_options=['predictor'],  # fix preditor lr
)

# lr scheduler
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
