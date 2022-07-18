_base_ = "r50_mixups_CE_none_4xb64.py"

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.1, 1,],  # list of alpha
    mix_mode=["mixup", "cutmix",],  # list of chosen mixup modes
    mix_prob=[0.5, 0.5,],  # list of applying probs (sum=1), None for random applying
    mix_repeat=1,  # times of repeating mixup aug
    head=dict(
        type='ClsMixupHead',  # mixup soft CE loss
        loss=dict(type='CrossEntropyLoss',  # soft CE (one-hot encoding)
            use_soft=True, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=True, two_hot=False,
        in_channels=2048, num_classes=1000)
)

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        save_interval=5004 * 10,  # plot every 10ep
        iter_per_epoch=5004),
]
