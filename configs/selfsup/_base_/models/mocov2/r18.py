# model settings
model = dict(
    type='MOCO',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='BN'),
        style='pytorch'),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=512, hid_channels=2048, out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2)
)
