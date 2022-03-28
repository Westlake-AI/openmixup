_base_ = '../../_base_/datasets/imagenet/mocov2_sz224_bs64.py'

# model settings
model = dict(
    type='SimSiam',
    backbone=dict(
        type='ResNet_mmcls',
        depth=50,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048, hid_channels=2048, out_channels=2048,
        num_layers=3,
        with_bias=True, with_last_bn=False, with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
                in_channels=2048, hid_channels=512, out_channels=2048,
                num_layers=2,
                with_avg_pool=False,
                with_bias=True, with_last_bn=False, with_last_bias=True))
)

# interval for accumulate gradient
update_interval = 1  # total: 4 x bs64 x 1 accumulates = bs256

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.05,  # lr=0.05 / bs256
    weight_decay=1e-4, momentum=0.9,
    paramwise_options={
        'predictor': dict(lr=0.05),  # fix preditor lr
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
# optimizer args
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16, grad_clip=None)

# additional lr scheduler (parawise_options required in optimizer)
addtional_scheduler = dict(
    policy='Fixed', paramwise_options=['predictor'],  # fix preditor lr
)

# lr scheduler
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
