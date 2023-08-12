_base_ = "../convnext_t_mixups_bs100.py"

# model settings
model = dict(
    alpha=2,
    mix_mode="manifoldmix",
    backbone=dict(
        type='ConvNeXt_Mix_CIFAR',
    ),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
