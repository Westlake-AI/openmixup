_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    backbone=dict(type='ResNetV1c'),
    decode_head=dict(num_classes=150), auxiliary_head=dict(num_classes=150))

# By default, models are trained on 4 GPUs with 4 images per GPU
data = dict(samples_per_gpu=4)

# mixed precision
fp16 = dict(loss_scale='dynamic')
