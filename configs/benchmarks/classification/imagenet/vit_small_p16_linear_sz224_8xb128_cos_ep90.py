_base_ = [
    '../_base_/models/vit_small_p16_linear.py',
    '../_base_/datasets/imagenet_sz224_4xbs64.py',
    '../_base_/default_runtime.py',
]
# MoCo v3 linear probing setting

# model settings
model = dict(backbone=dict(frozen_stages=12, norm_eval=True))

# data
data = dict(imgs_per_gpu=128, workers_per_gpu=8)  # total 128*8=1024, 8 GPU linear cls

use_fp16 = True

# optimizer
optimizer = dict(type='SGD', lr=12, momentum=0.9, weight_decay=0.)

# learning policy
lr_config = dict(policy='step', step=[60, 80])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=90)
