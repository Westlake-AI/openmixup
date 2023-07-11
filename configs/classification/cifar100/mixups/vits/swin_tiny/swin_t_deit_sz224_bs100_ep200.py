_base_ = "../swin_t_mixups_sz224_bs100.py"

# model settings
model = dict(
    alpha=[1, 0.8],
    mix_mode=['cutmix', 'mixup'],
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
