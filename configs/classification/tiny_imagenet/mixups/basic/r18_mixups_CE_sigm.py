_base_ = "r18_mixups_CE_none.py"

# model settings
model = dict(
    head=dict(
        type='ClsMixupHead',  # mixup CE loss
        loss=dict(type='CrossEntropyLoss',  # BCE sigmoid (one-hot encoding)
            use_soft=False, use_sigmoid=True, loss_weight=1.0),
        with_avg_pool=True, multi_label=True, two_hot=False, two_hot_scale=1,
        in_channels=512, num_classes=200)
)
