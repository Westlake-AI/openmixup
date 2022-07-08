_base_ = [
    '../../_base_/models/byol/r18.py',
    '../../_base_/datasets/imagenet/byol_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# dataset settings for SSL metrics
val_data_source_cfg = dict(type='ImageNet')
# ImageNet dataset for SSL metrics
val_data_train_list = 'data/meta/ImageNet/train_labeled_full.txt'
val_data_train_root = 'data/ImageNet/train'
val_data_test_list = 'data/meta/ImageNet/val_labeled.txt'
val_data_test_root = 'data/ImageNet/val/'

val_test_pipeline = [
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
        pipeline=val_test_pipeline,
        prefetch=False,
    ),
    val=dict(
        type='ClassificationDataset',
        data_source=dict(
            list_file=val_data_test_list, root=val_data_test_root,
            **val_data_source_cfg),
        pipeline=val_test_pipeline,
        prefetch=False,
    ))

# interval for accumulate gradient
update_interval = 8  # total: 8 x bs64 x 8 accumulates = bs4096

# additional hooks
custom_hooks = [
    dict(type='CosineScheduleHook',  # update momentum
        end_momentum=1.0,
        adjust_scope=[0.01, 1.0],
        warming_up="constant",
        interval=update_interval),
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

# optimizer
optimizer = dict(
    type='LARS',
    lr=7.2,  # lr=7.2 / bs4096 only for 100ep
    momentum=0.9, weight_decay=1e-6,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True),
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale='dynamic')
# optimizer args
optimizer_config = dict(update_interval=update_interval, grad_clip=None)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=0.,
    warmup='linear',
    warmup_iters=10, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
