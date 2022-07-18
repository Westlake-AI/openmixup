_base_ = "r50_rsb_a3_CE_none_4xb512_fp16.py"

# model settings
model = dict(
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='CrossEntropyLoss',  # mixup soft CE (one-hot encoding)
            use_soft=True, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=True, two_hot=False,
        in_channels=2048, num_classes=1000)
)
