_base_ = '../../../../base.py'
# model settings
model = dict(
    type='SelfTuning',
    queue_size=32,
    proj_dim=1024,
    class_num=200,
    momentum=0.999,
    temperature=0.07,
    pretrained="work_dirs/my_pretrains/official/resnet18_pytorch.pth",
    backbone=dict(
        type='ResNet_mmcls',
        depth=18,
        in_channels=3,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='BN'),
    ),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=512,
        hid_channels=512,
        out_channels=1024,
        with_avg_pool=True),
    head_cls=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=512, num_classes=196)
)
# dataset settings
data_source_cfg = dict(type='ImageNet')
# StanfordCars
data_base = "data/StanfordCars/"
data_train_labeled_list = data_base + 'image_list/train_30.txt'  # download from Self-Tuning
data_train_unlabeled_list = data_base + 'image_list/unlabeled_30.txt'
data_train_root = data_base
data_test_list = data_base + 'image_list/test.txt'
data_test_root = data_base

dataset_type = 'SemiSupervisedDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
train_pipeline = [
    dict(type='Resize', size=256),
    dict(type='RandomResizedCrop', size=224, scale=(0.08, 1.)),
    dict(type='RandomHorizontalFlip'),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
]

# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
test_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=24,
    workers_per_gpu=4,
    drop_last=True,  # moco
    train=dict(
        type=dataset_type,
        data_source_labeled=dict(
            list_file=data_train_labeled_list, root=data_train_root, **data_source_cfg),
        data_source_unlabeled=dict(
            list_file=data_train_unlabeled_list, root=data_train_root, **data_source_cfg),
        pipeline_labeled=train_pipeline,
        pipeline_unlabeled=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type='ClassificationDataset',
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
optimizer = dict(type='SGD',
                lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=False,
                paramwise_options={'head_cls': dict(lr_mult=10)})  # classification head
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=400)

# runtime settings
total_epochs = 120
