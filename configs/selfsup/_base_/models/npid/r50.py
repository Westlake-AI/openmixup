# model settings
model = dict(
    type='NPID',
    neg_num=65536,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    neck=dict(
        type='LinearNeck',
        in_channels=2048, out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.07),
    memory_bank=dict(
        type='SimpleMemory', length=1281167, feat_dim=128, momentum=0.5)
)
