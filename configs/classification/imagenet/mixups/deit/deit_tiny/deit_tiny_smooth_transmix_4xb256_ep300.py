_base_ = "../deit_tiny_smooth_mix_4xb256.py"

# model settings
model = dict(
    alpha=[0.1, 1.0],
    mix_mode=["mixup", "transmix"],  # using TransMix instead of CutMix
    mix_prob=[0.2, 0.8],
    backbone=dict(return_attn=True),  # return the attn map for TransMix
)

# optimizer
optimizer = dict(
    weight_decay=0.03,  # for faster convergence
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
