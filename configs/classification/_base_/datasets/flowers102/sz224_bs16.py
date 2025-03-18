# dataset settings
data_source_cfg = dict(type='ImageNet')
# Flowers102
# data_train_list = 'data/meta/Flowers102/train.txt'
# data_train_root = 'data/Flowers102/train/'
# data_test_list = 'data/meta/Flowers102/valid.txt'
# data_test_root = 'data/Flowers102/valid/'
data_train_list = 'data/meta/Oxford_Flowers102/train_labeled.txt'
data_train_root = 'data/Oxford_Flowers102/jpg/'
data_test_list = 'data/meta/Oxford_Flowers102/test_labeled.txt'
data_test_root = 'data/Oxford_Flowers102/jpg/'


dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='Resize', size=256),
    dict(type='RandomResizedCrop', size=224, scale=[0.5, 1.0]),
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
    imgs_per_gpu=16,  # small batch size for fine-grained
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
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
