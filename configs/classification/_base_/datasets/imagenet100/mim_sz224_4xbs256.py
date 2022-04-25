_base_ = '../imagenet/mim_sz224_4xbs256.py'

# dataset settings
data_source_cfg = dict(type='ImageNet')
# ImageNet dataset, 100 class
data_train_list = 'data/meta/ImageNet100/train_labeled.txt'
data_train_root = 'data/ImageNet/train'
data_test_list = 'data/meta/ImageNet100/val_labeled.txt'
data_test_root = 'data/ImageNet/val/'

# dataset summary
data = dict(
    imgs_per_gpu=256,
    workers_per_gpu=8,
    train=dict(
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
    ),
    val=dict(
        data_source=dict(
            list_file=data_test_list, root=data_test_root,
            **data_source_cfg),
    ),
)

# validation hook
evaluation = dict(
    initial=False,
    interval=1,
    imgs_per_gpu=100,
    workers_per_gpu=2,
    eval_param=dict(topk=(1, 5)))
