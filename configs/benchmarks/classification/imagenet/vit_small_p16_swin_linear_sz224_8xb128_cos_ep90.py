_base_ = [
    '../_base_/models/vit_small_p16.py',
    '../_base_/datasets/imagenet_swin_sz224_8xbs128.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(backbone=dict(frozen_stages=12, norm_eval=True))

use_fp16 = True

# optimizer
optimizer = dict(type='SGD', lr=12, momentum=0.9, weight_decay=0.)

# learning policy
lr_config = dict(policy='step', step=[60, 80])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=90)
