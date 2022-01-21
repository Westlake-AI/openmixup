_base_ = '../../../../base.py'
# pre_conv_cfg or value_neck_cfg
conv1x1=dict(
    type="ConvNeck",
    in_channels=1024, hid_channels=512, out_channels=1,  # MixBlock v
    num_layers=2, kernel_size=1,
    with_last_bn=False, norm_cfg=dict(type='BN'),  # ori
    with_last_dropout=0, with_avg_pool=False, with_residual=False)  # no res

# model settings
model = dict(
    type='AutoMixup_V2_V0824',  # v08.24
    pretrained=None,
    alpha=2.0,
    momentum=0.999,
    mask_layer=2,
    mask_loss=0.1,  # using loss
    mask_adjust=0,
    pre_one_loss=-1,  # v08.24 pre loss
    pre_mix_loss=-1,  # v08.24 pre loss
    lam_margin=0.08,
    debug=False,
    backbone=dict(
        type='ResNeXt_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(2,3),
        groups=32, width_per_group=4,
        style='pytorch'),
    mix_block = dict(  # V2, update v08.24
        type='PixelMixBlock_V2_v0824',
        in_channels=1024, reduction=2, use_scale=True, double_norm=False,
        attention_mode='embedded_gaussian', unsampling_mode='bilinear',
        pre_norm_cfg=None,  # v08.27
        pre_conv_cfg=None,  # v08.23
        pre_attn_cfg=None,  # v08.23
        pre_neck_cfg=None,  # v08.24, SimCLR neck
        pre_head_cfg=None,  # v08.24, SimCLR head
        lam_concat=False, lam_concat_v=False,  # V1: none
        lam_mul=True, lam_residual=True, lam_mul_k=0.25,  # V2 lam: mult + k=0.25
        value_neck_cfg=conv1x1,  # v08.29
        x_qk_concat=True, x_v_concat=False,  # V2 x concat: q,k
        mask_loss_mode="L1+Variance", mask_loss_margin=0.1,  # L1 + var loss
        mask_mode="none_v_",
        frozen=False),
    head_one=dict(
        type='ClsHead',  # CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=2048, num_classes=200),
    head_mix=dict(
        type='ClsMixupHead',  # mixup, CE
        loss=dict(type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, two_hot=False, in_channels=2048, num_classes=200),
    head_weights=dict(  # 07.25
        head_mix_q=1, head_one_q=1, head_mix_k=1, head_one_k=1),
)

# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=False,
    mclient_path='/mnt/lustre/share/memcached_client')
# tiny imagenet
data_train_list = 'data/TinyImagenet200/meta/train_labeled.txt'  # train 10w
data_train_root = 'data/TinyImagenet200/train/'
data_test_list = 'data/TinyImagenet200/meta/val_labeled.txt'  # val 1w
data_test_root = 'data/TinyImagenet200/val/'

dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=64),
    dict(type='RandomHorizontalFlip'),
]
test_pipeline = [
    dict(type='Resize', size=64),
]
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
test_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=50,
    workers_per_gpu=4,
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
            list_file=data_test_list, root=data_test_root,
            **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False,
    ),
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
        end_momentum=0.99999,
        adjust_scope=[0.1, 1.0],
        warming_up="constant",
        interval=1),
    dict(type='SAVEHook',
        save_interval=20000,  # 20 ep
        iter_per_epoch=1000,
    )
]

# optimizer
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001,
            paramwise_options={'mix_block': dict(lr=0.2)})  # required parawise_option
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.05)  # min_lr=5e-2
checkpoint_config = dict(interval=800)

# runtime settings
total_epochs = 400
