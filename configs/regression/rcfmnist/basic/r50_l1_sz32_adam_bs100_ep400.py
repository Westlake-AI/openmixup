_base_ = [
    '../../_base_/datasets/rcfmnist/sz32_bs100.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='ResNet_CIFAR',  # CIFAR version
        depth=50,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style='pytorch'),
    head=dict(
        type='RegHead',
        loss=dict(type='RegressionLoss', mode="l1_loss", loss_weight=1.0),
        with_avg_pool=True, in_channels=2048, out_channels=1)
)

# optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-6)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)
