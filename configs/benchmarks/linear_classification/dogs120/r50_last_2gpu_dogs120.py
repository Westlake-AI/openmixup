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
        num_classes=120))  # Dogs-120
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/mnt/lustre/share/memcached_client')
# test: Dogs-120 dataset
base = "/usr/commondata/public/Dogs120/"
data_train_list = base + 'classification_meta_0/train_labeled.txt'  # Dogs-120 labeled train, 100 per class, 12000
data_train_root = base + "Images"
data_test_list = base + 'classification_meta_0/test_labeled.txt'  # Dogs-120 labeled test, 100 per class
data_test_root = base + "Images"
# resize setting
resizeto = 224
dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet
train_pipeline = [
    dict(type='RandomResizedCrop', size=resizeto),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=resizeto),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    # imgs_per_gpu=32,  # total 32*8=256, 8GPU linear cls
    # workers_per_gpu=12,
    imgs_per_gpu=128,  # total 128*2=256, 2GPU linear cls
    workers_per_gpu=10,
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
        initial=True,
        interval=10, # 1,
        imgs_per_gpu=128,
        workers_per_gpu=8,  # 4,
        eval_param=dict(topk=(1, 5)))
]
# optimizer
optimizer = dict(type='SGD', lr=30., momentum=0.9, weight_decay=0.)  # ImageNet basic lr
# optimizer = dict(type='SGD', lr=1.0, momentum=0.9, weight_decay=0.)  # STL-10 lr
# learning policy
lr_config = dict(
    policy='step',
    step=[60, 80]
    # step=[30, 40]
    # step=[18, 24]
)
checkpoint_config = dict(interval=50)
# runtime settings
total_epochs = 100
# total_epochs = 50

# * 1205: Dogs-120, baseline, size=224, try ImageNet basic lr=30.0
# Test: CUDA_VISIBLE_DEVICES=4,5 PORT=25105 bash benchmarks/dist_train_linear.sh configs/benchmarks/linear_classification/dogs120/r50_last_2gpu_dogs120.py ./work_dirs/my_pretrains/
