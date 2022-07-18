_base_ = "r50_rsb_a3_CE_none_4xb512_fp16.py"

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.1, 1, 0.2,],  # list of alpha
    mix_mode=["mixup", "cutmix", "manifoldmix"],  # list of chosen mixup modes
    mix_prob=[1/3, 1/3, 1/3],  # list of applying probs (sum=1), None for random applying
    mix_repeat=1,  # times of repeating mixup aug
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='CrossEntropyLoss',  # mixup BCE loss (one-hot encoding)
            use_soft=False, use_sigmoid=True, loss_weight=1.0),
        with_avg_pool=True, multi_label=True, two_hot=False,
        in_channels=2048, num_classes=1000)
)
