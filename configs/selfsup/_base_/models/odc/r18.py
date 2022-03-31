# model settings
model = dict(
    type='ODC',
    with_sobel=False,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    neck=dict(
        type='ODCNeck',
        in_channels=512, hid_channels=512, out_channels=256,
        with_avg_pool=True),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=False,  # already has avgpool in the neck
        in_channels=256, num_classes=10000),
    memory_bank=dict(
        type='ODCMemory',
        length=1281167, feat_dim=256, momentum=0.5,
        num_classes=10000, min_cluster=20, debug=False)
)
