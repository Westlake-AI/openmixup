_base_ = [
    '../../../_base_/datasets/aircrafts/per_30_sz224_bs24.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='SelfTuning',
    queue_size=32,
    proj_dim=1024,
    class_num=200,
    momentum=0.999,
    temperature=0.07,
    pretrained="work_dirs/my_pretrains/official/resnet50_pytorch.pth",
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='BN'),
    ),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=1024,
        with_avg_pool=True),
    head_cls=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=2048, num_classes=100)
)

# optimizer
optimizer = dict(type='SGD',
                lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=False,
                paramwise_options={'head_cls': dict(lr_mult=10)})  # classification head
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=150)
