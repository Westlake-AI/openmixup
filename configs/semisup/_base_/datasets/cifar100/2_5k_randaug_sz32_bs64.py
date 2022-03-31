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
        data_source_labeled=dict(  # 2.5k labeled
            split='train', return_label=True, num_labeled=2500, **data_source_cfg),
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

# validation hook
evaluation = dict(
    initial=False,
    interval=5,
    imgs_per_gpu=100,
    workers_per_gpu=4,
    eval_param=dict(topk=(1, 5)))

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
