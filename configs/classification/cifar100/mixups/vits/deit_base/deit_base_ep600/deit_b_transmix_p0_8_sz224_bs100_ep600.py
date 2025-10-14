_base_ = "../../deit_b_mixups_sz224_bs100.py"

# model settings
model = dict(
    alpha=[0.8, 1.0],
    mix_mode=["mixup", "transmix"],  # using TransMix instead of CutMix
    mix_prob=[0.2, 0.8],
    backbone=dict(return_attn=True),  # return the attn map for TransMix
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=600)
