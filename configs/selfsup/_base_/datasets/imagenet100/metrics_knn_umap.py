# dataset settings for SSL metrics
val_data_source_cfg = dict(type='ImageNet', return_label=False)
# ImageNet dataset, 100 class
val_data_train_list = 'data/meta/ImageNet100/train_labeled.txt'
val_data_train_root = 'data/ImageNet/train'
val_data_test_list = 'data/meta/ImageNet100/val_labeled.txt'
val_data_test_root = 'data/ImageNet/val/'

test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
val_data = dict(
    train=dict(
        type='ClassificationDataset',
        data_source=dict(
            list_file=val_data_train_list, root=val_data_train_root,
            **val_data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False,
    ),
    val=dict(
        type='ClassificationDataset',
        data_source=dict(
            list_file=val_data_test_list, root=val_data_test_root,
            **val_data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False,
    ))

# additional hooks
custom_hooks = [
    dict(type='SSLMetricHook',
        val_dataset=val_data['val'],
        train_dataset=val_data['train'],  # remove it if metric_mode is None
        forward_mode='vis',
        metric_mode=['knn', 'svm',],  # linear metric (take a bit long time on ImageNet-100)
        metric_args=dict(
            knn=20, temperature=0.07, chunk_size=256,
            dataset='onehot', costs_list="0.01,0.1,1.0,10.0,100.0", default_cost=None, num_workers=8,),
        visual_mode='umap',  # 'tsne' or 'umap'
        visual_args=dict(n_epochs=300, plot_backend='seaborn'),
        save_val=False,  # whether to save results
        initial=True,
        interval=25,
        imgs_per_gpu=256,
        workers_per_gpu=6,
        eval_param=dict(topk=(1, 5))),
]
