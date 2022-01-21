_base_ = '../../../base.py'
# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=1,
    mix_mode="mixup",
    mix_args=dict(
        manifoldmix=dict(layer=(0, 3)),
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
        fmix=dict(decay_power=3, size=(64,64), max_soft=0., reformulate=False)
    ),
    backbone=dict(  # v09.13 manifoldmix
        type='ResNet_Mix_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    head=dict(
        type='ClsMixupHead',  # BCE version of CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=True, loss_weight=1.0),
        with_avg_pool=True, multi_label=True,
        two_hot=False, two_hot_mode='none', two_hot_thr=1, two_hot_idx=1, two_hot_scale=1,
        in_channels=512, num_classes=200)
)
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/mnt/lustre/share/memcached_client')
# tiny imagenet
data_train_list = 'data/TinyImagenet200/meta/train_labeled.txt'  # train 10w
data_train_root = 'data/TinyImagenet200/train/'
data_test_list = 'data/TinyImagenet200/meta/val_labeled.txt'  # val 1w
data_test_root = 'data/TinyImagenet200/val/'

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet (used in prefetch)
train_pipeline = [
    dict(type='RandomResizedCrop', size=64),
    dict(type='RandomHorizontalFlip'),
]
test_pipeline = []
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
test_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=100,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False,
    ))
# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=False,
        interval=1,
        imgs_per_gpu=100,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5)))
]
# optimizer
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# lr scheduler
lr_config = dict(policy='CosineAnnealing', min_lr=0)
checkpoint_config = dict(interval=800)

# runtime settings
total_epochs = 400
