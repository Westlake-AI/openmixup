_base_ = "../deit_b_mixups_sz224_bs100.py"

# model settings
model = dict(
    alpha=0.8,
    mix_mode="mixup",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
