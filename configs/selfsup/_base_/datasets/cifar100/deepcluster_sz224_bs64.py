_base_ = '../cifar10/deepcluster_sz224_bs64.py'

# dataset settings
<<<<<<< HEAD
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
=======
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
data_source_cfg = dict(type='CIFAR100', root='data/cifar100/')

# dataset summary
data = dict(
    train=dict(
        data_source=dict(split='train', return_label=False, **data_source_cfg),
    ),
    extract=dict(
        type="ExtractDataset",
        data_source=dict(split='train', return_label=False, **data_source_cfg),
    ),
)

# additional hooks
num_classes = 1000
custom_hooks = [
    dict(
        type='DeepClusterHook',
        extractor=dict(
            imgs_per_gpu=256,
            workers_per_gpu=8,
            dataset=data['extract'],
            prefetch=False,
            img_norm_cfg=img_norm_cfg),
        clustering=dict(type='Kmeans', k=num_classes, pca_dim=256),
        unif_sampling=True,
        reweight=False,
        reweight_pow=0.5,
        initial=True,  # call initially
        interval=1)
]
