# dataset settings
data_source_cfg = dict(type='CIFAR100', root='data/cifar100/')

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=[0.2, 1], interpolation=3),  # bicubic
    dict(type='RandomHorizontalFlip'),
]
test_pipeline = [
    dict(type='Resize', size=256, interpolation=3),  # 0.85
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=64,  # V100: 64 x 4gpus = bs256
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False),
)

# validation hook
evaluation = dict(
    initial=False,
    interval=1,
    imgs_per_gpu=100,
    workers_per_gpu=4,
    eval_param=dict(topk=(1, 5)))

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
