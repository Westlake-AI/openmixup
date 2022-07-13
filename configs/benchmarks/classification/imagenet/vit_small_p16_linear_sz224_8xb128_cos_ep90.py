_base_ = [
    '../_base_/models/vit_small_p16.py',
    '../_base_/datasets/imagenet_swin_sz224_8xbs128.py',
    '../_base_/default_runtime.py',
]

# MoCo v3 linear probing setting

# model settings
model = dict(
    backbone=dict(frozen_stages=12, norm_eval=True),
    head=dict(
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=100, mode='original', loss_weight=1.0),
        num_classes=100))

# data
data = dict(imgs_per_gpu=128, workers_per_gpu=8)  # total 128*8=1024, 8 GPU linear cls

# optimizer
optimizer = dict(type='SGD', lr=12, momentum=0.9, weight_decay=0.)

# learning policy
lr_config = dict(policy='step', step=[60, 80])

# apex
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=1)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=90)
