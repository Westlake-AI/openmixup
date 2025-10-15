_base_ = "../deit_s_mixups_sz224_bs100_randaug.py"

# model settings
model = dict(
    alpha=[1, 0.8],
    mix_mode=['cutmix', 'mixup'],
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
