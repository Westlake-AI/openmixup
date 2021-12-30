_base_ = '../../../base.py'
# model settings
model = dict(
    type='Classification',
    pretrained=None,
    with_sobel=False,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=4),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=10))  # to cifar-10
# dataset settings
data_source_cfg = dict(type='Cifar10', root='./data/cifar10/')
dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet for transfer
# img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])  # cifar-10
resizeto = 64
train_pipeline = [
    dict(type='RandomResizedCrop', size=resizeto, scale=[0.2, 1.0]),
    # dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=resizeto),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    # imgs_per_gpu=64,
    # workers_per_gpu=8,
    imgs_per_gpu=128,
    workers_per_gpu=10,
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
        initial=True,
        # initial=False,
        interval=10,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5)))
]
# optimizer
# optimizer = dict(type='SGD', lr=30., momentum=0.9, weight_decay=0.)  # imagenet MoCo version
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.)  # imagenet
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    step=[60, 80]
)
checkpoint_config = dict(interval=50)
# runtime settings
total_epochs = 100

# * 1230: Transfer test from STL10 to CIFAR-10 (linear evaluation), size=64, bs128
# Test: CUDA_VISIBLE_DEVICES=3 PORT=25717 bash benchmarks/dist_train_linear_1gpu.sh configs/benchmarks/linear_classification/cifar/r50_last_1gpu_cifar10_from_stl10_lr01.py ./work_dirs/my_pretrains/
