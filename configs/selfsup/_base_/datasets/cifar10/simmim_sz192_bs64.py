# dataset settings
data_source_cfg = dict(type='CIFAR10', root='data/cifar10/')

dataset_type = 'ExtractDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
train_pipeline = [
    dict(type='RandomResizedCrop', size=192, scale=(0.67, 1.0), ratio=(3. / 4., 4. / 3.)),
    dict(type='RandomHorizontalFlip')
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
train_pipeline.append(
    dict(type='BlockwiseMaskGenerator',
        input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6,
        mask_color='zero', mask_only=False),
    )

# dataset summary
data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', return_label=False, **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch))

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
