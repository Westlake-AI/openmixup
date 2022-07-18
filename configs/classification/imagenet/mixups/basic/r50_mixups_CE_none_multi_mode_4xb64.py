_base_ = "r50_mixups_CE_none_4xb64.py"

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.1, 1, 1,],  # list of alpha
    mix_mode=["mixup", "cutmix", "vanilla",],  # list of chosen mixup modes
    mix_prob=None,  # list of applying probs (sum=1), None for random applying
    mix_repeat=1,  # times of repeating mixup aug
)
