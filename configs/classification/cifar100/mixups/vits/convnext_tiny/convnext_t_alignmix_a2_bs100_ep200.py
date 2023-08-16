_base_ = "../convnext_t_mixups_bs100.py"

# model settings
model = dict(
    alpha=[2, 2],  # list of alpha
    mix_mode=["mixup", "alignmix"],  # AlignMix switches to {'mixup' or 'manifoldmix'}
    backbone=dict(gap_before_final_norm=False),
    head=dict(with_avg_pool=True),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
