_base_ = '../../../base.py'
# model settings
model = dict(
    type='Classification',
    pretrained=None,
    with_sobel=False,
    backbone=dict(  # mmclassification
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        frozen_stages=4,
    ),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=512,
        num_classes=100))  # cifar-100
# dataset settings
data_source_cfg = dict(type='Cifar100', root='./data/cifar100/')
dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])  # cifar-10 for cifar100

train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    # imgs_per_gpu=64,
    # workers_per_gpu=8,
    imgs_per_gpu=128,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline))

# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        # initial=True,
        initial=False,
        interval=10,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5)))
]
# optimizer
# optimizer = dict(type='SGD', lr=30., momentum=0.9, weight_decay=0.)  # imagenet MoCo version
optimizer = dict(type='SGD', lr=1.0, momentum=0.9, weight_decay=0.)  # imagenet, [choosed]
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    # step=[60, 80],  # 2 step
    step=[30, 60, 80],  # 3 step
    gamma=0.2,
)
checkpoint_config = dict(interval=50)
# runtime settings
total_epochs = 100

# * 1224: CIFAR-10 linear evaluation, size=32, bs128. try lr=1.0 + 3 steps
# Test: CUDA_VISIBLE_DEVICES=0 PORT=25917 bash benchmarks/dist_train_linear_1gpu.sh configs/benchmarks/linear_classification/cifar100/r18_last_1gpu_cifar100.py ./work_dirs/my_pretrains/
