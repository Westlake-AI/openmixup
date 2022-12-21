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
data_source_cfg = dict(type='ImageNet')
# ImageNet dataset
data_train_list = 'data/meta/ImageNet/train_labeled_full.txt'
data_train_root = 'data/ImageNet/train'
data_test_list = 'data/meta/ImageNet/val_labeled.txt'
data_test_root = 'data/ImageNet/val/'

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),  # bicubic
    dict(type='RandomHorizontalFlip'),
    dict(type='RandAugment',
        policies=rand_increasing_policies,
        num_policies=2, total_level=10,
        magnitude_level=7, magnitude_std=0.5,
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
    imgs_per_gpu=256,  # V100/A100: 256 x 8gpus x 1 accumulate = bs2048
    workers_per_gpu=10,  # according to total cpus cores, usually 4 workers per 32~128 imgs
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False,
    ))

# validation hook
evaluation = dict(
    initial=False,
    interval=1,
    imgs_per_gpu=256,
    workers_per_gpu=6,
    eval_param=dict(topk=(1, 5)))

# checkpoint
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
