# Refers to `_RAND_INCREASING_TRANSFORMS` in pytorch-image-models
rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(type='SolarizeAdd', magnitude_key='magnitude', magnitude_range=(0, 110)),
    dict(type='ColorTransform', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(type='Brightness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(type='Shear',
        magnitude_key='magnitude', magnitude_range=(0, 0.3), direction='horizontal'),
    dict(type='Shear',
        magnitude_key='magnitude', magnitude_range=(0, 0.3), direction='vertical'),
    dict(type='Translate',
        magnitude_key='magnitude', magnitude_range=(0, 0.45), direction='horizontal'),
    dict(type='Translate',
        magnitude_key='magnitude', magnitude_range=(0, 0.45), direction='vertical'),
]

_base_ = '../../../../base.py'
# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(
        manifoldmix=dict(layer=(0, 3)),
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
        fmix=dict(decay_power=3, size=(224,224), max_soft=0., reformulate=False)
    ),
    backbone=dict(
        type='TIMMBackbone',
        model_name='deit_small_patch16_224',
        features_only=True,  # remove head in timm
        pretrained=False,
        checkpoint_path=None,
        in_channels=3,
    ),
    head=dict(
        type='ClsMixupHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=False,  # no gap in ViT
        in_channels=384, num_classes=1000)
)
# dataset settings
data_source_cfg = dict(type='ImageNet')
# ImageNet dataset
data_train_list = 'data/meta/ImageNet/train_labeled_full.txt'
data_train_root = 'data/ImageNet/train'
data_test_list = 'data/meta/ImageNet/val_labeled.txt'
data_test_root = 'data/ImageNet/val/'
# Notice: Though official DeiT settings use `RepeatAugment`, we achieve competitive performances
#   without it. This repo removes `RepeatAugment`.
sampler = "DistributedSampler"

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3), # bicubic
    dict(type='RandomHorizontalFlip'),
    dict(type='RandAugment',
        policies=rand_increasing_policies,
        num_policies=2, total_level=10,
        magnitude_level=9, magnitude_std=0.5,  # DeiT
        hparams=dict(
            pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing_numpy',  # before ToTensor and Normalize
        erase_prob=0.25,
        mode='rand', min_area_ratio=0.02, max_area_ratio=1 / 3,
        fill_color=[104, 116, 124], fill_std=[58, 57, 57]),  # RGB
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
    imgs_per_gpu=256,  # V100/A100: 256 x 4gpus x 1 accumulate = bs1024
    workers_per_gpu=8,  # according to total cpus cores, usually 4 workers per 32~128 imgs
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
# additional hooks
update_interval = 1  # interval for accumulate gradient
custom_hooks = [
    dict(type='ValidateHook',
        dataset=data['val'],
        initial=False,
        interval=2,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5))),
    dict(type='EMAHook',  # EMA_W = (1 - m) * EMA_W + m * W
        momentum=0.99996,
        warmup='linear',
        warmup_iters=20 * 1252, warmup_ratio=0.9,  # warmup 20 epochs.
        update_interval=update_interval,
    )
]
# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,  # lr = 5e-4 * (256 * 4) * 1 accumulate / 512 = 1e-3 / bs1024
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
    })
# apex
use_fp16 = False
# Notice: official ViT (DeiT or Swim) settings don't apply use_fp16=True. This repo use
#   use_fp16=True for fast training and better performances (around +0.1%).
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0),  # DeiT and Swim repos suggest max_norm=5.0
    update_interval=update_interval, use_fp16=use_fp16)

# lr scheduler: Swim for DeiT
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,  # {5e-6, 1e-6} yields similar performances
    warmup='linear',
    warmup_iters=20, warmup_by_epoch=True,  # warmup 20 epochs.
    warmup_ratio=1e-5,
)
checkpoint_config = dict(interval=100)

# runtime settings
total_epochs = 300
