# FixMatch RandAugment policy for CIFAR
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

# dataset settings
data_source_cfg = dict(type='CIFAR100', root='data/cifar100/')

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=[0.8, 1], interpolation=3),  # bicubic
    dict(type='RandomHorizontalFlip'),
    dict(type='RandAugment',
        policies=fixmatch_augment_policies,
        num_policies=2, total_level=10,
        magnitude_level=7, magnitude_std=0.5,  # 'rand-m7-mstd0.5'
        hparams=dict(
            pad_val=[114, 123, 125], interpolation='bicubic')),
]
test_pipeline = [
    dict(type='Resize', size=224, interpolation=3),
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
test_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=100,  # 100 x 1gpu = 100
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False),
)

# validation hook
evaluation = dict(
    initial=False,
    interval=1,
    imgs_per_gpu=100,
    workers_per_gpu=4,
    eval_param=dict(topk=(1, 5)),
    )

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
