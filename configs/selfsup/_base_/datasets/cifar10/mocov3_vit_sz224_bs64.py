# dataset settings
data_source_cfg = dict(type='CIFAR10', root='data/cifar10/')

dataset_type = 'MultiViewDataset'
<<<<<<< HEAD
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
=======
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
# the difference between ResNet50 and ViT pipeline is the `scale` in
# `RandomResizedCrop`, `scale=(0.08, 1.)` in ViT pipeline
train_pipeline1 = [
    dict(type='RandomResizedCrop', size=224, scale=(0.08, 1.)),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomAppliedTrans',
        transforms=[dict(
            type='ColorJitter',
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=1.),
    dict(type='Solarization', p=0.),
]
train_pipeline2 = [
    dict(type='RandomResizedCrop', size=224, scale=(0.08, 1.)),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomAppliedTrans',
        transforms=[
            dict(type='ColorJitter',
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.1),
    dict(type='Solarization', p=0.2),
]

# prefetch
prefetch = True
if not prefetch:
    train_pipeline1.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
    train_pipeline2.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    imgs_per_gpu=64,  # V100: 64 x 8gpus x 8 accumulates = bs4096
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', return_label=False, **data_source_cfg),
        num_views=[1, 1],
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch,
    ))

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
