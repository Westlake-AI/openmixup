_base_ = '../../../../base.py'
# model settings
model = dict(
    type='Classification',
    pretrained=None,
    with_sobel=False,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=4),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=512, num_classes=10))  # stl 10
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/mnt/lustre/share/memcached_client')
# test: STL-10 dataset
data_train_list = 'data/stl10/meta/train_5k_labeled.txt'  # stl10 labeled 5k train
data_train_root = 'data/stl10/train/'  # using labeled train set
data_test_list = 'data/stl10/meta/test_8k_labeled.txt'  # stl10 labeled 8k test
data_test_root = 'data/stl10/test/'  # using labeled test set
# resize setting
dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet
train_pipeline = [
    dict(type='RandomResizedCrop', size=96),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=96),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=256,  # total 256, 1GPU linear cls
    workers_per_gpu=4,
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
        interval=2,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5)))
]
# optimizer
# optimizer = dict(type='SGD', lr=30., momentum=0.9, weight_decay=0.)  # MoCo ImageNet
optimizer = dict(type='SGD', lr=1.0, momentum=0.9, weight_decay=0.)  # [OK]
# learning policy
lr_config = dict(
    policy='step',
    step=[60, 80]
)
checkpoint_config = dict(interval=200)
# runtime settings
total_epochs = 100

# * STL-10, baseline, size=96, lr=1.0
# Test: CUDA_VISIBLE_DEVICES=6 PORT=25519 bash benchmarks/dist_train_linear_1gpu.sh configs/benchmarks/linear_classification/stl10/r18_last_1gpu_stl10.py []
