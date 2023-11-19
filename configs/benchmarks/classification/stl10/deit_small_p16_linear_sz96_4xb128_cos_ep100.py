_base_ = [
    '../_base_/models/deit_small_p16.py',
    '../_base_/datasets/stl10_swin_ft_sz96_8xbs128.py',
    '../_base_/default_runtime.py',
]

# MoCo v3 linear probing setting

# model settings
model = dict(
    backbone=dict(frozen_stages=12, norm_eval=True),
    head=dict(
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=10, mode='original', loss_weight=1.0),
        num_classes=10))

# data
data = dict(imgs_per_gpu=128, workers_per_gpu=8)  # total 128*4=512, 4 GPU linear cls

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# fp16
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=1, grad_clip=None)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
