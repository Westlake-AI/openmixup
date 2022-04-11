# dataset settings
data_source_cfg = dict(type='CIFAR10', root='data/cifar10/')

dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomHorizontalFlip'),
    dict(type='ColorJitter',
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
]

# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    imgs_per_gpu=64,  # V100: 64 x 4gpus = bs256
    workers_per_gpu=6,  # according to total cpus cores, usually 4 workers per 32~128 imgs
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', return_label=False, **data_source_cfg),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ))

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
