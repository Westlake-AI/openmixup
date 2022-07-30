_base_ = "vit_base_dec8_dim512_8xb128_accu4_cos_fp16_ep1600.py"

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)
