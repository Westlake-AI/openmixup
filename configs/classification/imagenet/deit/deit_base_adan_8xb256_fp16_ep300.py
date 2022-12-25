_base_ = './deit_base_adan_8xb256_fp16_ep150.py'

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
