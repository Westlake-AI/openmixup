_base_ = "../deit_s_mixups_sz224_bs100_randaug.py"

# model settings
model = dict(
    alpha=0.8,
    mix_mode="mixup",
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
