# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='ResNet_CIFAR',  # CIFAR version
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        frozen_stages=-1,
        style='pytorch'),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=1000)
)
