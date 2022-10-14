# dataset settings
data_source_cfg = dict(type='ImageNet')
# ImageNet dataset
data_train_list = 'data/meta/ImageNet/train_labeled_full.txt'
data_train_root = 'data/ImageNet/train'
data_test_list = 'data/meta/ImageNet/val_labeled.txt'
data_test_root = 'data/ImageNet/val/'

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),  # bicubic
    dict(type='RandomHorizontalFlip'),
]
test_pipeline_1 = [
    dict(type='Resize', size=256, interpolation=3),  # 0.85
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline_2 = [
    dict(type='Resize', size=256, interpolation=3),  # 0.85
    dict(type='RandomVerticalFlip', p=0.5),
    dict(type='PlaceCrop', size=224, start=[0, 5, 10, 15,]),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type="MultiViewDataset",  # use multi-view for test time augmentations
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        num_views=[2, 4],
        pipelines=[test_pipeline_1, test_pipeline_2],
        prefetch=False,
    ))

# validation hook
evaluation = dict(
    initial=False,
    interval=1,
    imgs_per_gpu=128,
    workers_per_gpu=4,
    eval_param=dict(topk=(1, 5)))

# checkpoint
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
