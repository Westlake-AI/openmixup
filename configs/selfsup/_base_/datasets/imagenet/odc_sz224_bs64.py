# dataset settings
data_source_cfg = dict(type='ImageNet', return_label=False)
# ImageNet dataset
data_train_list = 'data/meta/ImageNet/train_full.txt'
data_train_root = 'data/ImageNet/train'
data_test_list = 'data/meta/ImageNet/val.txt'
data_test_root = 'data/ImageNet/val/'

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
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
    extract_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    imgs_per_gpu=64,  # V100: 64 x 8gpus = bs512
    workers_per_gpu=6,  # according to total cpus cores, usually 4 workers per 32~128 imgs
    sampling_replace=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch),
    extract=dict(
        type="ExtractDataset",
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=extract_pipeline,
        prefetch=prefetch),
)

# additional hooks
num_classes = 10000
custom_hooks = [
    dict(
        type='DeepClusterHook',
        extractor=dict(
            imgs_per_gpu=256,
            workers_per_gpu=8,
            dataset=data['extract'],
            prefetch=prefetch,
            img_norm_cfg=img_norm_cfg),
        clustering=dict(type='Kmeans', k=num_classes, pca_dim=-1),  # no pca
        unif_sampling=False,
        reweight=True,
        reweight_pow=0.5,
        init_memory=True,
        initial=True,  # call initially
        interval=9999999999),  # initial only
    dict(
        type='ODCHook',
        centroids_update_interval=10,  # iter
        deal_with_small_clusters_interval=1,
        evaluate_interval=50,
        reweight=True,
        reweight_pow=0.5)
]

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
