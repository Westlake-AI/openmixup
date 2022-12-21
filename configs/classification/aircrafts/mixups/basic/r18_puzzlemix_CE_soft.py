_base_ = "r18_mixups_CE_none.py"

# model settings
model = dict(
    alpha=1,
    mix_mode="puzzlemix",
    head=dict(
        type='ClsMixupHead',  # mixup soft CE loss
        loss=dict(type='CrossEntropyLoss',  # soft CE (one-hot encoding)
            use_soft=True, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=True, two_hot=False,
        in_channels=512, num_classes=100)
)
