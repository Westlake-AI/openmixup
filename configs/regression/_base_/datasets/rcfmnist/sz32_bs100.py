# dataset settings
data_source_cfg = dict(type='RCFMNIST', root='data/')

dataset_type = 'RegressionDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
train_pipeline = [
    dict(type='Resize', size=32),
    dict(type='RandomCrop', size=32, padding=2),
    # dict(type='RandomHorizontalFlip'),
]
test_pipeline = [
    dict(type='Resize', size=32),
]
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
test_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=100,  # 100 x 1gpu = 100
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
    eval_param=dict(
        metric=['mse', 'mae', 'rmse', 'mape'],
        metric_options=dict(average_model='mean'),
    save_best='mse')
)

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
