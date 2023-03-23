_base_ = "vit_large_dec8_dim512_8xb256_accu2_cos.py"

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)
