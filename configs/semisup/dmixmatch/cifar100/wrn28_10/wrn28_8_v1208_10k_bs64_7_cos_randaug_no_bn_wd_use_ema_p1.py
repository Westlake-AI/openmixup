_base_ = '../../../../base.py'

# FixMatch RandAugment policy (14)
fixmatch_augment_policies = [
    dict(type='AutoContrast'),
    dict(type='Brightness', magnitude_key='magnitude', magnitude_range=(0.05, 0.95)),
    dict(type='ColorTransform', magnitude_key='magnitude', magnitude_range=(0.05, 0.95)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0.05, 0.95)),
    dict(type='Equalize'),
    dict(type='Identity'),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 8)),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
    dict(type='Sharpness', magnitude_key='magnitude', magnitude_range=(0.05, 0.95)),
    dict(type='Shear', magnitude_key='magnitude', magnitude_range=(0, 0.3), direction='horizontal'),
    dict(type='Shear', magnitude_key='magnitude', magnitude_range=(0, 0.3), direction='vertical'),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(type='Translate', magnitude_key='magnitude', magnitude_range=(0, 0.30), direction='horizontal'),
    dict(type='Translate', magnitude_key='magnitude', magnitude_range=(0, 0.30), direction='vertical')
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
        depth=28,
        widen_factor=8,  # WRN-28-8, 128-256-512
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
# dataset settings
data_source_cfg = dict(type='Cifar100', root='data/cifar100/')

dataset_type = 'SemiSupervisedDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
train_weak_pipeline = [
    dict(type='RandomCrop', size=32, padding=4, padding_mode="reflect"),  # tricks
    dict(type='RandomHorizontalFlip'),
]
train_strong_pipeline = [
    dict(type='RandomCrop', size=32, padding=4, padding_mode="reflect"),  # tricks
    dict(type='RandomHorizontalFlip'),
    dict(type='RandAugment',  # TorchSSL num_policies=3
        policies=fixmatch_augment_policies,
        num_policies=3, magnitude_level=30),  # using full magnitude
    dict(type='RandomAppliedTrans',  # TorchSSL + cutout p=0.5
        transforms=[
            dict(type='Cutout', shape=4, pad_val=(125, 123, 114))  # 32 * 0.5
        ], p=0.5),
]
test_pipeline = []
# prefetch
prefetch = True
if not prefetch:
    train_weak_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
    train_strong_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
test_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=64,  # (7x64 + 1x64) x 1gpus = 512
    workers_per_gpu=6,
    drop_last=True,  # moco
    train=dict(
        type=dataset_type,
        data_source_labeled=dict(  # 10k labeled
            split='train', return_label=True, num_labeled=10000, **data_source_cfg),
        data_source_unlabeled=dict(  # unlabeled
            split='train', return_label=False, num_labeled=None, **data_source_cfg),
        pipeline_labeled=train_weak_pipeline,
        pipeline_unlabeled=train_weak_pipeline,
        pipeline_strong=train_strong_pipeline,
        ret_samples=dict(x_l_2=False, x_ul_2=True),  # x_l, x_ul_w, x_ul_s
        prefetch=prefetch,
    ),
    val=dict(
        type='ClassificationDataset',
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False,
    ))

# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=False,
        interval=1,
        imgs_per_gpu=100,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5))),
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
checkpoint_config = dict(interval=1600)

# runtime settings
total_epochs = 800
