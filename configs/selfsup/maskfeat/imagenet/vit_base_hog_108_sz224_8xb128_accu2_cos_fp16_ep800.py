_base_ = "../vit_base_hog_108_sz224_8xb128_accu2_cos_fp16_ep300.py"

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=800)
