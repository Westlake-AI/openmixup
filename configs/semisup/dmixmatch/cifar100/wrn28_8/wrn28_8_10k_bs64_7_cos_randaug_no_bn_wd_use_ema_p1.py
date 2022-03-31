_base_ = [
    '../../../_base_/datasets/cifar100/10k_randaug_sz32_bs64.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='DMixMatch',
    momentum=0.999,
    temperature=0.5,
    p_cutoff=0.95,
    weight_ul=1.0,
    hard_label=True,
    ratio_ul=7,
    ema_pseudo=1.0,
    pretrained=None,
    alpha=1,
    mix_mode="mixup",
    label_rescale='labeled',
    mix_args=dict(
        manifoldmix=dict(layer=(0, 2)),  # WRN
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
        fmix=dict(decay_power=3, size=(32,32), max_soft=0., reformulate=False)
    ),
    backbone=dict(
        type='WideResNet',
        first_stride=1,  # CIFAR
        in_channels=3,
        depth=28, widen_factor=8,  # WRN-28-8, 128-256-512
        drop_rate=0.,
        out_indices=(2,),
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=100),
    head_mix=dict(
        type='ClsMixupHead',  # soft CE decoupled mixup
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0,
            use_soft=True, use_sigmoid=False, use_mix_decouple=True,  # try decouple mixup CE
        ),
        with_avg_pool=True, multi_label=True, two_hot=False, two_hot_scale=1,  # try two-hot
        lam_scale_mode='pow', lam_thr=1, lam_idx=1,  # lam rescale, default 'pow'
        eta_weight=dict(eta=1, mode="both", thr=0),  # eta for decouple mixup
        in_channels=512, num_classes=100),
    loss_weights=dict(
        decent_weight=[],
        accent_weight=['weight_mix_lu'],
        weight_one=1, weight_mix_ll=1, weight_mix_lu=1),
)

# additional hooks
custom_hooks = [
    dict(type='CosineScheduleHook',
        end_momentum=0.99999,
        restart_step=1e11,  # never
        adjust_scope=[0.1, 1.0],
        warming_up="constant",
        interval=1),
    dict(type='CustomCosineAnnealingHook',  # basic 'cos_annealing'
        attr_name="cos_annealing", attr_base=1, by_epoch=False,  # by iter
        min_attr=0,)
]

# optimizer
optimizer = dict(type='SGD',
                lr=0.03, momentum=0.9, weight_decay=0.001, nesterov=False,
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
runner = dict(type='EpochBasedRunner', max_epochs=800)
