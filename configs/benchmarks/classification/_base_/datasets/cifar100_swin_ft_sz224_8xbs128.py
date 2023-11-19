# Refers to `_RAND_INCREASING_TRANSFORMS` in pytorch-image-models
rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(type='SolarizeAdd', magnitude_key='magnitude', magnitude_range=(0, 110)),
    dict(type='ColorTransform', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(type='Brightness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(type='Shear',
        magnitude_key='magnitude', magnitude_range=(0, 0.3), direction='horizontal'),
    dict(type='Shear',
        magnitude_key='magnitude', magnitude_range=(0, 0.3), direction='vertical'),
    dict(type='Translate',
        magnitude_key='magnitude', magnitude_range=(0, 0.45), direction='horizontal'),
    dict(type='Translate',
        magnitude_key='magnitude', magnitude_range=(0, 0.45), direction='vertical'),
]

# dataset settings
data_source_cfg = dict(type='CIFAR100', root='data/cifar100/')

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3, scale=[0.2, 1]),  # bicubic
    dict(type='RandomHorizontalFlip'),
    dict(type='RandAugment',
        policies=rand_increasing_policies,
        num_policies=2, total_level=10,
        magnitude_level=9, magnitude_std=0.5,  # DeiT or Swin
        hparams=dict(
            pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing_numpy',  # before ToTensor and Normalize
        erase_prob=0.25,
        mode='rand', min_area_ratio=0.02, max_area_ratio=1 / 3,
        fill_color=[104, 116, 124],
        fill_std=[58, 57, 58]),
]
test_pipeline = [
    dict(type='Resize', size=256, interpolation=3),  # 0.85
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=128,
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
    eval_param=dict(topk=(1, 5)))

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
