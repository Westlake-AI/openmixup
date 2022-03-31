# model settings
model = dict(
    type='SwAV',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    neck=dict(
        type='SwAVNeck',
        in_channels=512, hid_channels=2048, out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='SwAVHead',
        feat_dim=128,  # equal to neck['out_channels']
        epsilon=0.05,
        temperature=0.1,
        num_crops=[2, 6],)
)
