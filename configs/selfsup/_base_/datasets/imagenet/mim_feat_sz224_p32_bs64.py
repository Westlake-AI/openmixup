# dataset settings
data_source_cfg = dict(type='ImageNet', return_label=False)
# ImageNet dataset
data_train_list = 'data/meta/ImageNet/train_full.txt'
data_train_root = 'data/ImageNet/train'

dataset_type = 'MaskedImageDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.5, 1.0), ratio=(3. / 4., 4. / 3.)),
    dict(type='RandomHorizontalFlip'),
]
train_mask_pipeline = [
    dict(type='BlockwiseMaskGenerator',
        input_size=224, mask_patch_size=32, model_patch_size=4, mask_ratio=0.4, mask_only=True,
    ),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    imgs_per_gpu=64,  # V100: 64 x 8gpus x 8 accumulates = bs4096
    workers_per_gpu=6,  # according to total cpus cores, usually 4 workers per 32~128 imgs
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        mask_pipeline=train_mask_pipeline,
        feature_mode='hog',
        feature_args=dict(orientations=9, pixels_per_cell=(16,16), cells_per_block=(1,1)),
        prefetch=prefetch))

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)
