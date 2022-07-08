_base_ = [
    '../../_base_/models/simsiam/r18.py',
    '../../_base_/datasets/imagenet/mocov2_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# dataset settings for SSL metrics
val_data_source_cfg = dict(type='ImageNet')
# ImageNet dataset for SSL metrics
val_data_train_list = 'data/meta/ImageNet/train_labeled_full.txt'
val_data_train_root = 'data/ImageNet/train'
val_data_test_list = 'data/meta/ImageNet/val_labeled.txt'
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
        metric_mode='knn',  # linear metric (take a bit long time on imagenet)
        metric_args=dict(knn=20, temperature=0.07, chunk_size=256),
        visual_mode='umap',  # 'tsne' or 'umap'
        visual_args=dict(n_epochs=200, plot_backend='seaborn'),
        save_val=False,  # whether to save results
        initial=True,
        interval=10,
        imgs_per_gpu=256,
        workers_per_gpu=6,
        eval_param=dict(topk=(1, 5))),
]

# interval for accumulate gradient
update_interval = 1  # total: 8 x bs64 x 1 accumulates = bs512

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.05 * 2,  # lr=0.05 / bs256
    weight_decay=1e-4, momentum=0.9,
    paramwise_options={
        'predictor': dict(lr=0.05 * 2,),  # fix preditor lr
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval, grad_clip=None)

# additional lr scheduler (parawise_options required in optimizer)
addtional_scheduler = dict(
    policy='Fixed', paramwise_options=['predictor'],  # fix preditor lr
)

# lr scheduler
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
