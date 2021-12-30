_base_ = '../../../base.py'
# model settings
model = dict(
    type='Representation', # 0802
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        # depth=18,
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

# test: Pets-37 dataset
base = "/usr/commondata/public/Pets37/"
data_train_list = base + 'classification_meta_0/train_labeled.txt'  # Pets-37 labeled train, 3680
data_train_root = base + "images"
data_test_list = base + 'classification_meta_0/test_labeled.txt'  # Pets-37 labeled test
data_test_root = base + "images"

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])  # coco2017

resizeto = 224
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=resizeto),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    # imgs_per_gpu=32,  # total 32*8=256, 8GPU linear cls
    imgs_per_gpu=128,
    workers_per_gpu=10,
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg
            # **data_source_cfg
        ),
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

# using for visualize representation of Pets-37
