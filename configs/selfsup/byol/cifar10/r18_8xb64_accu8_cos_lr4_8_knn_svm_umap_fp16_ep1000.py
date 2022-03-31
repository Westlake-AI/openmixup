_base_ = [
    '../../_base_/models/byol/r18.py',
    '../../_base_/datasets/cifar10/byol_sz224_bs256.py',
    '../../_base_/default_runtime.py',
]

# dataset settings for SSL metrics
data_source_cfg = dict(type='CIFAR10', root='data/cifar10/')
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
val_data = dict(
    train=dict(
        type='ClassificationDataset',
        data_source=dict(split='train', **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False,
    ),
    val=dict(
        type='ClassificationDataset',
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False,
    ))

# interval for accumulate gradient
update_interval = 2  # total: 8 x bs256 x 2 accumulates = bs4096

# additional hooks
custom_hooks = [
    dict(type='CosineScheduleHook',  # update momentum
        end_momentum=1.0,
        adjust_scope=[0.01, 1.0],
        warming_up="constant",
        interval=update_interval),
    dict(type='SSLMetricHook',
        val_dataset=val_data['train'],
        train_dataset=val_data['val'],  # remove it if metric_mode is None
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

# optimizer
optimizer = dict(
    type='LARS',
    lr=4.8,  # lr=4.8 / bs4096 for longer training
    momentum=0.9, weight_decay=1e-6,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True),
    })

# apex
use_fp16 = True
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
# optimizer args
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16, grad_clip=None)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=0.,
    warmup='linear',
    warmup_iters=10, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# log, 50k / 4096
log_config = dict(interval=10)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1000)
