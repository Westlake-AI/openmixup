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
    neck=dict(type='AvgPoolNeck'),  # 7x7x2048 -> 2048
)
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/mnt/lustre/share/memcached_client')
# test: STL-10 dataset
data_train_list = 'data/STL/stl10/meta/train_5k_labeled.txt'  # stl10 labeled 5k train
data_train_root = 'data/STL/stl10/train/'  # using labeled train set
data_test_list = 'data/STL/stl10/meta/test_8k_labeled.txt'  # stl10 labeled 8k test
data_test_root = 'data/STL/stl10/test/'  # using labeled test set

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])  # coco2017
resizeto = 96
test_pipeline = [
    dict(type='Resize', size=resizeto),
    dict(type='CenterCrop', size=resizeto),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    # imgs_per_gpu=32,  # total 32*8=256, 8GPU linear cls
    imgs_per_gpu=128,
    workers_per_gpu=4,
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

# using for visualize representation of STL-10
