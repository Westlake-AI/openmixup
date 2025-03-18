# dataset settings
data_source_cfg = dict(type='CIFAR10', root='data/cifar10/')

dataset_type = 'MultiViewDataset'
<<<<<<< HEAD
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
=======
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomAppliedTrans',
        transforms=[dict(
            type='ColorJitter',
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),
]

# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    imgs_per_gpu=64,  # V100: 64 x 4gpus = bs256
<<<<<<< HEAD
    workers_per_gpu=6,  # according to total cpus cores, usually 4 workers per 32~128 imgs
=======
    workers_per_gpu=4,  # according to total cpus cores, usually 4 workers per 32~128 imgs
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
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
