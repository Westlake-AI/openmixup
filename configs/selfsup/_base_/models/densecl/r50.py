# model settings
model = dict(
    type='DenseCL',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    loss_lambda=0.5,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='BN'),
        style='pytorch'),
    neck=dict(
        type='DenseCLNeck',
        in_channels=2048, hid_channels=2048, out_channels=128,
        num_grid=None),
    head=dict(type='ContrastiveHead', temperature=0.2)
)
