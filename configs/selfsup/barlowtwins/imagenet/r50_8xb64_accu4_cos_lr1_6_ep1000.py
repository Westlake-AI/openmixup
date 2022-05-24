_base_ = 'r50_8xb64_accu4_cos_lr4_8.py'

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1000)
