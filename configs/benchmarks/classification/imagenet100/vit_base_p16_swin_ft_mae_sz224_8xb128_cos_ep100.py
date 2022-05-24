_base_ = '../imagenet/vit_base_p16_swin_ft_sz224_8xb128_cos_ep100.py'

# model settings
model = dict(
    head=dict(
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=100, mode='original', loss_weight=1.0),
        num_classes=100))

# dataset settings
data_source_cfg = dict(type='ImageNet')
# ImageNet dataset
data_train_list = 'data/meta/ImageNet100/train_labeled.txt'
data_train_root = 'data/ImageNet/train'
data_test_list = 'data/meta/ImageNet100/val_labeled.txt'
data_test_root = 'data/ImageNet/val/'

data = dict(
    train=dict(
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
    ),
    val=dict(
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
    ))
