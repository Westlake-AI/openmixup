_base_ = 'r18_8xb64_step_ep1000.py'

# dataset settings for SSL metrics
val_data_source_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_source_cfg = dict(type='CIFAR100', root='data/cifar100/')

val_test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
]
extract_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
]

# prefetch
prefetch = False  # should be false for metrics
if not prefetch:
    val_test_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **val_data_source_cfg)])
    extract_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **val_data_source_cfg)])

val_data = dict(
    train=dict(
        type='ClassificationDataset',
        data_source=dict(split='train', **data_source_cfg),
        pipeline=val_test_pipeline,
        prefetch=False,
    ),
    val=dict(
        type='ClassificationDataset',
        data_source=dict(split='test', **data_source_cfg),
        pipeline=val_test_pipeline,
        prefetch=False,
    ),
    extract=dict(
        type="ExtractDataset",
        data_source=dict(split='train', return_label=False, **data_source_cfg),
        pipeline=extract_pipeline,
        prefetch=prefetch),
)

# additional hooks
num_classes = 1000
custom_hooks = [
    dict(type='DeepClusterHook',
        extractor=dict(
            imgs_per_gpu=256,
            workers_per_gpu=8,
            dataset=val_data['extract'],
            prefetch=prefetch,
            img_norm_cfg=val_data_source_cfg),
        clustering=dict(type='Kmeans', k=num_classes, pca_dim=256),
        unif_sampling=True,
        reweight=False,
        reweight_pow=0.5,
        initial=True,  # call initially
        interval=1),
    dict(type='SSLMetricHook',
        val_dataset=val_data['val'],
        train_dataset=val_data['train'],  # remove it if metric_mode is None
        forward_mode='vis',
        metric_mode=['knn', 'svm',],  # linear metric (take a bit long time on imagenet)
        metric_args=dict(
            knn=200, temperature=0.07, chunk_size=256,
            dataset='onehot', costs_list="0.01,0.1,1.0,10.0,100.0", default_cost=None, num_workers=8,),
        visual_mode='umap',  # 'tsne' or 'umap'
        visual_args=dict(n_epochs=400, plot_backend='seaborn'),
        save_val=False,  # whether to save results
        initial=True,
        interval=50,
        imgs_per_gpu=256,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5))),
]
