_base_ = '../../../../base.py'
# value_neck_cfg
conv1x1=dict(
    type="ConvNeck",
    in_channels=256, hid_channels=128, out_channels=1,  # MixBlock v
    num_layers=2, kernel_size=1,
    with_last_bn=False, norm_cfg=dict(type='BN'),  # default
    with_last_dropout=0, with_avg_pool=False, with_residual=False)  # no res & dropout

# model settings
model = dict(
    type='AutoMixup',
    pretrained=None,
    alpha=2.0,
    momentum=0.999,  # 0.999 to 0.999999
    mask_layer=2,
    mask_loss=0.1,  # using mask loss
    mask_adjust=0,  # prob of adjusting bb mask in terms of lam by mixup, 0.25 for CIFAR
    lam_margin=0.08,  # degenerate to mixup when lam or 1-lam <= 0.08
    mask_up_override=None,  # If not none, override upsampling when train MixBlock
    debug=True,  # show attention and content map
    backbone=dict(
        type='ResNet_CIFAR',  # CIFAR version
        depth=18,
        num_stages=4,
        out_indices=(2,3),  # stage-3 for MixBlock, x-1: stage-x
        style='pytorch'),
    mix_block = dict(  # SAMix
        type='PixelMixBlock',
        in_channels=256, reduction=2, use_scale=True, double_norm=False,
        attention_mode='embedded_gaussian',
        unsampling_mode=['bilinear',],  # str or list, tricks in SAMix
        lam_concat=False, lam_concat_v=False,  # AutoMix.V1: none
        lam_mul=True, lam_residual=True, lam_mul_k=-1,  # SAMix lam: mult + k=-1 (optional 0.25)
        value_neck_cfg=conv1x1,  # SAMix: non-linear value
        x_qk_concat=True, x_v_concat=False,  # SAMix x concat: q,k
        # att_norm_cfg=dict(type='BN'),  # norm after q,k (design for fp16, also conduct better performace in fp32)
        mask_loss_mode="L1+Variance", mask_loss_margin=0.1,  # L1+Var loss, tricks in SAMix
        mask_mode="none_v_",
        frozen=False),
    head_one=dict(
        type='ClsHead',  # soft CE
        loss=dict(type='CrossEntropyLoss', use_soft=True, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=True, in_channels=512, num_classes=200),
    head_mix=dict(  # backbone
        type='ClsMixupHead',  # mixup, soft CE
        loss=dict(type='CrossEntropyLoss', use_soft=True, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=True, in_channels=512, num_classes=200),
    head_mix_k=dict(  # mixblock
        type='ClsMixupHead',  # mixup, soft CE (onehot encoding)
        loss=dict(type='CrossEntropyLoss', use_soft=True, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=True,
        neg_weight=0.5,  # try neg (eta in SAMix)
        in_channels=512, num_classes=100),
    head_weights=dict(
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)
# dataset settings
data_source_cfg = dict(type='ImageNet')
# Tiny Imagenet
data_train_list = 'data/TinyImageNet/meta/train_labeled.txt'  # train 10w
data_train_root = 'data/TinyImageNet/train/'
data_test_list = 'data/TinyImageNet/meta/val_labeled.txt'  # val 1w
data_test_root = 'data/TinyImageNet/val/'

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=64),
    dict(type='RandomHorizontalFlip'),
]
test_pipeline = []
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
test_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=100,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False)
)

# additional hooks
custom_hooks = [
    dict(type='ValidateHook',
        dataset=data['val'],
        initial=False,
        interval=1,
        imgs_per_gpu=100,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, 5))),
    dict(type='CosineScheduleHook',
        end_momentum=0.999999,
        adjust_scope=[0.1, 1.0],
        warming_up="constant",
        interval=1),
    dict(type='SAVEHook',
        iter_per_epoch=1000,
        save_interval=25000,  # plot every 500 x 25 ep
    )
]

# optimizer
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001,
            paramwise_options={'mix_block': dict(lr=0.1)})  # required parawise_option
# fp16
use_fp16 = False
# optimizer args
optimizer_config = dict(update_interval=1, use_fp16=use_fp16, grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing', min_lr=5e-2)  # adjust mlr for small-scale datasets
checkpoint_config = dict(interval=400)

# additional scheduler
addtional_scheduler = dict(
    policy='CosineAnnealing', min_lr=1e-3,  # 0.1 x 1/100
    paramwise_options=['mix_block'],
)

# runtime settings
total_epochs = 400
