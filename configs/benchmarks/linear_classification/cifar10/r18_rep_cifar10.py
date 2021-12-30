_base_ = '../../../base.py'
# model settings
model = dict(
    type='Representation',
    pretrained=None,
    backbone=dict(  # mmclassification
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    neck=dict(type='AvgPoolNeck'),
)
# dataset settings
data_source_cfg = dict(type='Cifar10', root='./data/cifar10/')
dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])  # cifar-10

test_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=256,
    workers_per_gpu=8,
    val=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline),
)
# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=True,
        interval=10,
        imgs_per_gpu=128,
        workers_per_gpu=8,
        eval_param=dict(topk=(1, 5)))
]
# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
# learning policy
lr_config = dict(policy='step', step=[150, 250])
checkpoint_config = dict(interval=50)
# runtime settings
total_epochs = 350

