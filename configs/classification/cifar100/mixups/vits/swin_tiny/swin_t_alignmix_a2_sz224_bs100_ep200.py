_base_ = "../swin_t_mixups_sz224_bs100.py"

# model settings
model = dict(
    alpha=[2, 2],  # list of alpha
    mix_mode=["mixup", "alignmix"],  # AlignMix switches to {'mixup' or 'manifoldmix'}
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
