# dataset settings
data_source_cfg = dict(type='ImageNet', return_label=False)
# ImageNet dataset
data_train_list = 'data/meta/STL10/train_10w_unlabeled.txt'
data_train_root = 'data/stl10/train/'
data_test_list = 'data/meta/STL10/test_8k_unlabeled.txt'
data_test_root = 'data/stl10/test/'

dataset_type = 'ExtractDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.67, 1.0), ratio=(3. / 4., 4. / 3.)),
    dict(type='RandomHorizontalFlip')
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
train_pipeline.append(
    dict(type='BlockwiseMaskGenerator',
        input_size=224, mask_patch_size=32, model_patch_size=16, mask_ratio=0.6,
        mask_color='zero', mask_only=False),
    )

# dataset summary
data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch))

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
