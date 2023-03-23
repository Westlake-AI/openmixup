_base_ = "vit_base_dec8_dim512_8xb512_cos.py"

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1600)
