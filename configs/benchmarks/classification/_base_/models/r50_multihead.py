# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=-1,
        style='pytorch'),
    head=dict(
        type='MultiClsHead',
        pool_type='specified',
        in_indices=(1, 2, 3, 4,),  # x: stage-x
        with_last_layer_unpool=False,
        backbone='resnet50',
        norm_cfg=dict(type='SyncBN', momentum=0.1, affine=False),
        num_classes=1000)
)
