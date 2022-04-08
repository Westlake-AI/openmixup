_base_ = '../cifar10/mocov3_vit_sz224_bs64.py'

# dataset settings
data_source_cfg = dict(type='CIFAR100', root='data/cifar100/')

# dataset summary
data = dict(
    train=dict(
        data_source=dict(split='train', return_label=False, **data_source_cfg),
    ))
