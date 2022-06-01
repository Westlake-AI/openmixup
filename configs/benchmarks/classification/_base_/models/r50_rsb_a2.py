# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.1, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_prob=None,  # None for random applying
    mix_args=dict(),
    backbone=dict(
        type='ResNet',  # normal
        depth=50,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN'),
        drop_path_rate=0.05,  # RSB A2
        style='pytorch'),
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='CrossEntropyLoss',  # mixup BCE loss (one-hot encoding)
            use_soft=False, use_sigmoid=True, loss_weight=1.0),
        with_avg_pool=True, multi_label=True, two_hot=False,
        in_channels=2048, num_classes=1000)
)
