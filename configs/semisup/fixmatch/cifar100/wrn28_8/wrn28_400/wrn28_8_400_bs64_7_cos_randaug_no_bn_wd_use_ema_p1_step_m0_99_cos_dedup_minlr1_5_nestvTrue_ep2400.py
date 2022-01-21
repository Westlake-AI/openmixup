_base_ = '../../../../../base.py'

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
        depth=28,
        widen_factor=8,  # WRN-28-8, 128-256-512
        drop_rate=0.,
        out_indices=(2,),
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=100)
)
# dataset settings
data_source_cfg = dict(type='CIFAR100', root='data/cifar100/')

dataset_type = 'SemiSupervisedDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
train_l_pipeline = [
    dict(type='RandomCrop', size=32, padding=4, padding_mode="reflect"),
    dict(type='RandomHorizontalFlip'),
]
train_weak_pipeline = [  # weak: raw test
    dict(type='RandomCrop', size=32, padding=2, padding_mode="reflect"),
    dict(type='RandomHorizontalFlip'),
]
train_strong_pipeline = [
    dict(type='RandomCrop', size=32, padding=4, padding_mode="reflect"),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandAugment',  # TorchSSL num_policies=3
        policies=fixmatch_augment_policies,
        num_policies=3, magnitude_level=30),  # using full magnitude
    dict(type='Cutout', shape=4, pad_val=(125, 123, 114)),  # TorchSSL stronger aug
]
test_pipeline = []
# prefetch
prefetch = True
if not prefetch:
    train_l_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
    train_weak_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
    train_strong_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
test_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=448,  # (7x64 + 1x64) x 1gpus = 512
    workers_per_gpu=8,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source_labeled=dict(  # 400 labeled
            split='train', return_label=True, num_labeled=400, **data_source_cfg),
        data_source_unlabeled=dict(  # unlabeled
            split='train', return_label=False, num_labeled=None, **data_source_cfg),
        pipeline_labeled=train_l_pipeline,
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
checkpoint_config = dict(interval=2400)

# runtime settings
total_epochs = 2400
