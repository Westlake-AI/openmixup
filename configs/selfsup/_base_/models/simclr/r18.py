# model settings
model = dict(
    type='SimCLR',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    neck=dict(
        type='NonLinearNeck',  # SimCLR non-linear neck
        in_channels=512, hid_channels=2048, out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.1)
)
