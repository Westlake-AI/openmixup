_base_ = '../../../base.py'
# model settings
model = dict(
    type='Representation',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=4),
    neck=dict(type='AvgPoolNeck'),
)
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/mnt/lustre/share/memcached_client')
# tiny imagenet
data_train_list = './data/TinyImagenet200/meta/train_unlabeled.txt'  # unlabeled train 10w
data_train_root = './data/TinyImagenet200/train/'
# data_test_list = './data/TinyImagenet200/meta/val_labeled.txt'  # val labeled 1w
# data_test_root = './data/TinyImagenet200/val/'
data_test_list = './data/TinyImagenet200/meta/train_20class_labeled.txt'  # val labeled 20 class
data_test_root = './data/TinyImagenet200/train/'

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet

resizeto = 64
test_pipeline = [
    dict(type='Resize', size=resizeto),
    dict(type='CenterCrop', size=resizeto),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=12,  # 5,
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root,
            **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=True,
        interval=1,
        imgs_per_gpu=128,
        workers_per_gpu=12,
        eval_param=dict(topk=(1, 5)))
]
# optimizer
optimizer = dict(type='SGD', lr=30., momentum=0.9, weight_decay=0.)
# learning policy
lr_config = dict(policy='step', step=[60, 80])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 100

# using for visualize representation of tiny-imagenet
