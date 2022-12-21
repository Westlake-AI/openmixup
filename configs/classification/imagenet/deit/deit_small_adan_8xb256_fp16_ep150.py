_base_ = [
    '../../_base_/models/deit/deit_small_p16_sz224.py',
    '../../_base_/datasets/imagenet/deit_adan_sz224_8xbs256.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    head=dict(
        type='VisionTransformerClsHead',  # mixup BCE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='multi_label', loss_weight=1.0),
        in_channels=384, num_classes=1000)
)

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=12)

# additional hooks
update_interval = 1  # 256 x 8gpus x 1 accumulates = bs2048

# optimizer
optimizer = dict(
    type='Adan',
    lr=1.5e-3,  # lr = 1.5e-3 / bs2048
    weight_decay=0.02, eps=1e-8, betas=(0.98, 0.92, 0.99),
    max_grad_norm=0.0,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
    })
                     
# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=60, warmup_by_epoch=True,  # warmup 60 epochs.
    warmup_ratio=1e-8,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=150)
