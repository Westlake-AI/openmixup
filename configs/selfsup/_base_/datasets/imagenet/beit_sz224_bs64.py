# dataset settings
data_source_cfg = dict(type='ImageNet', return_label=False)
# ImageNet dataset
data_train_list = 'data/meta/ImageNet/train_full.txt'
data_train_root = 'data/ImageNet/train'
data_test_list = 'data/meta/ImageNet/val.txt'
data_test_root = 'data/ImageNet/val/'

dataset_type = 'BEiTDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline1 = [
    dict(type='RandomHorizontalFlip'),
    dict(type='ColorJitter',
         brightness=0.4, contrast=0.4, saturation=0.4, hue=0.),
    dict(type='RandomResizedCrop', size=224, interpolation=2),  # bicubic
]
train_pipeline2 = [
    dict(type='RandomHorizontalFlip'),
    dict(type='ColorJitter',
         brightness=0.4, contrast=0.4, saturation=0.4, hue=0.),
    dict(type='RandomResizedCrop', size=112, interpolation=1),  # lanczos
]
train_mask_pipeline = [
    dict(type='BEiTMaskGenerator',
        input_size=(14, 14), num_masking_patches=75,
        max_num_patches=None, min_num_patches=16,
        mask_only=True),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline1.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
    train_pipeline2.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipelines=[train_pipeline1, train_pipeline2],
        mask_pipeline=train_mask_pipeline,
        prefetch=prefetch))

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
