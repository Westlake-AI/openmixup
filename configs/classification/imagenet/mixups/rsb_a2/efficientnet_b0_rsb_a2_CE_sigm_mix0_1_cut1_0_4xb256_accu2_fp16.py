_base_ = [
    '../../../_base_/datasets/imagenet/rsb_a2_sz224_8xbs256.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.1, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_prob=None,  # None for random applying
    mix_args=dict(
        manifoldmix=dict(layer=(0, 3)),
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
        fmix=dict(decay_power=3, size=(224,224), max_soft=0., reformulate=False)
    ),
    backbone=dict(
        type='EfficientNet',
        arch='b0',
        out_indices=(6,),  # x-1: stage-x
        norm_cfg=dict(type='BN', eps=1e-3),
    ),
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='CrossEntropyLoss',  # mixup BCE loss (one-hot encoding)
            use_soft=False, use_sigmoid=True, loss_weight=1.0),
        with_avg_pool=True, multi_label=True, two_hot=False,
        in_channels=1280, num_classes=1000)
)

# interval for accumulate gradient
update_interval = 2  # 256 x 4gpus x 2 accumulates = bs2048

# optimizer
optimizer = dict(type='LAMB', lr=0.005, weight_decay=0.02,  # RSB A2
                 paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
                    'bias': dict(weight_decay=0.)})
# apex
use_fp16 = True
# Notice: official RSB settings require use_fp16=True. We find use_fp16=True
#   produces better performance (+0.1% to +0.5%) than use_fp16=False.
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
