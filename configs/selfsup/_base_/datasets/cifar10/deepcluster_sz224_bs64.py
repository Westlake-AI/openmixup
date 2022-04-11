# dataset settings
data_source_cfg = dict(type='CIFAR10', root='data/cifar10/')

dataset_type = 'DeepClusterDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomRotation', degrees=2),
    dict(type='ColorJitter',
        brightness=0.4, contrast=0.4, saturation=1.0, hue=0.5),
    dict(type='RandomGrayscale', p=0.2),
]
extract_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
    extract_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    imgs_per_gpu=64,  # V100: 64 x 8gpus = bs512
    workers_per_gpu=6,  # according to total cpus cores, usually 4 workers per 32~128 imgs
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', return_label=False, **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch),
    extract=dict(
        type="ExtractDataset",
        data_source=dict(split='train', return_label=False, **data_source_cfg),
        pipeline=extract_pipeline,
        prefetch=prefetch),
)

# additional hooks
num_classes = 100
custom_hooks = [
    dict(
        type='DeepClusterHook',
        extractor=dict(
            imgs_per_gpu=256,
            workers_per_gpu=8,
            dataset=data['extract'],
            prefetch=prefetch,
            img_norm_cfg=img_norm_cfg),
        clustering=dict(type='Kmeans', k=num_classes, pca_dim=256),
        unif_sampling=True,
        reweight=False,
        reweight_pow=0.5,
        initial=True,  # call initially
        interval=1)
]

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
