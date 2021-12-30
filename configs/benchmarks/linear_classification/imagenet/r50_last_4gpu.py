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
        type='ClsHead', with_avg_pool=True, in_channels=2048, num_classes=1000))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/mnt/lustre/share/memcached_client')
# ImageNet dataset
data_root="data/ImageNet/"
data_train_list = 'data/meta/imagenet/train_labeled_full.txt'
data_train_root = data_root + 'train'
data_test_list = 'data/meta/imagenet/val_labeled.txt'
data_test_root = data_root + "val"

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    # imgs_per_gpu=32,  # total 32*8=256, 8GPU linear cls
    # workers_per_gpu=12,
    imgs_per_gpu=64,  # total 64*4=256
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=False,
        interval=1,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5)))
]
# optimizer
optimizer = dict(type='SGD', lr=30., momentum=0.9, weight_decay=0.)
# learning policy
lr_config = dict(
    policy='step',
    step=[60, 80]
    # step=[30, 40]
)
checkpoint_config = dict(interval=100)
# runtime settings
total_epochs = 100
# total_epochs = 50

# * 1105: ELIS-1030, baseline, size=224
# Test: CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=25012 bash benchmarks/dist_train_linear_4gpu.sh configs/benchmarks/linear_classification/imagenet/r50_last_4gpu.py ./work_dirs/my_pretrains/
