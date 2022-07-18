_base_ = "r50_mixups_CE_none_4xb64.py"

# model settings
model = dict(
    head=dict(
        type='ClsMixupHead',  # mixup CE loss
        loss=dict(type='CrossEntropyLoss',  # BCE sigmoid (one-hot encoding)
            use_soft=False, use_sigmoid=True, loss_weight=1.0),
        with_avg_pool=True, multi_label=True, two_hot=False, two_hot_scale=1,
        in_channels=2048, num_classes=1000)
)

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        save_interval=5004 * 10,  # plot every 10ep
        iter_per_epoch=5004),
]
