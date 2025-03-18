# model settings
model = dict(
    type='BarlowTwins',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    neck=dict(
        type='NonLinearNeck',
<<<<<<< HEAD
        in_channels=2048, hid_channels=8192, out_channels=8192,
=======
        in_channels=512, hid_channels=8192, out_channels=8192,
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        num_layers=3,
        with_bias=True, with_last_bn=False, with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type='LatentCrossCorrelationHead',
        in_channels=8192)
)
