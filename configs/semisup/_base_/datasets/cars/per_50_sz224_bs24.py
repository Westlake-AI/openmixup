# dataset settings
data_source_cfg = dict(type='ImageNet')
# StanfordCars
data_train_labeled_list = 'data/meta/Cars/image_list/train_50.txt'  # download from Self-Tuning
data_train_unlabeled_list = 'data/meta/Cars/image_list/unlabeled_50.txt'
data_train_root = 'data/StanfordCars/'
data_test_list = 'data/meta/Cars/image_list/test.txt'
data_test_root = 'data/StanfordCars/'

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
    imgs_per_gpu=24,  # 24 x 1gpu = 24
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

# validation hook
evaluation = dict(
    initial=False,
    interval=1,
    imgs_per_gpu=100,
    workers_per_gpu=4,
    eval_param=dict(topk=(1, 5)))

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
