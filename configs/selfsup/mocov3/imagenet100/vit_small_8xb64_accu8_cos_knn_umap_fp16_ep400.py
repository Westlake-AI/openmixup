_base_ = [
    '../../_base_/models/mocov3/vit_small.py',
    '../../_base_/datasets/imagenet100/mocov3_vit_sz224_bs64.py',
    '../../_base_/default_runtime.py',
]

# dataset settings for SSL metrics
val_data_source_cfg = dict(type='ImageNet')
# ImageNet dataset for SSL metrics
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

# interval for accumulate gradient
update_interval = 8  # total: 8 x bs64 x 8 accumulates = bs4096

# additional hooks
custom_hooks = [
    dict(type='CosineScheduleHook',  # update momentum
        end_momentum=1.0,
        adjust_scope=[0.05, 1.0],
        warming_up="constant",
        interval=update_interval),
    dict(type='SSLMetricHook',
        val_dataset=val_data['train'],
        train_dataset=val_data['val'],  # remove it if metric_mode is None
        forward_mode='vis',
        metric_mode='knn',  # linear metric (take a bit long time on imagenet)
        metric_args=dict(knn=20, temperature=0.07, chunk_size=128),
        visual_mode='umap',  # 'tsne' or 'umap'
        visual_args=dict(n_epochs=300, plot_backend='seaborn'),
        save_val=False,  # whether to save results
        initial=True,
        interval=50,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5))),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2.4e-3,  # bs4096
    betas=(0.9, 0.95), weight_decay=0.1,
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.)
    })

# apex
use_fp16 = True
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
# optimizer args
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16, grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)
