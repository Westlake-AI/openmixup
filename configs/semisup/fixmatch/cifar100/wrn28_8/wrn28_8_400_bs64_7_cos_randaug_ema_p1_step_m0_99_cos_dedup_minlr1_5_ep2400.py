_base_ = [
    '../../../_base_/datasets/cifar100/400_randaug_sz32_bs64.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='FixMatch',
    momentum=0.90,  # .90 to .999
    temperature=0.5,
    p_cutoff=0.95,
    weight_ul=1.0,
    hard_label=True,
    ratio_ul=7,
    ema_pseudo=1.0,  # 1.0 to 0, prob in [0, 1]
    deduplicate=True,
    pretrained=None,
    backbone=dict(
        type='WideResNet',
        first_stride=1,  # CIFAR version
        in_channels=3,
        depth=28, widen_factor=8,  # WRN-28-8, 128-256-512
        drop_rate=0.,
        out_indices=(2,),
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=100)
)

# additional hooks
custom_hooks = [
    dict(type='CosineScheduleHook',
        end_momentum=0.9999,
        restart_step=1e11,  # never restart
        adjust_scope=[0.50, 1.0],
        warming_up="constant",
        interval=1),
    dict(type='CustomStepHook',  # adjusting the prob to use ema_teacher
        attr_name="ema_pseudo", attr_base=1.0, min_attr=0,
        step=[300, 600, 900,], gamma=0.1, by_epoch=True,  # ep2400
    ),
]

# optimizer
optimizer = dict(type='SGD',
                lr=0.03, momentum=0.9, weight_decay=0.001, nesterov=True,
                paramwise_options={  # no wd for bn & bias
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.,),
                    'bias': dict(weight_decay=0.,)
                })
optimizer_config = dict(grad_clip=None)

# learning policy
# the original CosineAnnealing num_cycles = 1/2
# TorchSSL, cos num_cycles = 7/16, i.e., min_lr -> 1/5 * base_lr
lr_config = dict(policy='CosineAnnealing', min_lr=0.03 * 0.2)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=2400)
