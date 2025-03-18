_base_ = [
    '../../../_base_/datasets/cifar100/sz32_bs100.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
<<<<<<< HEAD
    alpha=1.0,
    mix_mode="cutmix",
=======
    alpha=1,
    mix_mode="mixup",
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    mix_args=dict(
        alignmix=dict(eps=0.1, max_iter=100),
        attentivemix=dict(grid_size=32, top_k=None, beta=8),  # AttentiveMix+ in this repo (use pre-trained)
        automix=dict(mask_adjust=0, lam_margin=0),  # require pre-trained mixblock
<<<<<<< HEAD
        fmix=dict(decay_power=0, size=(32,32), max_soft=0., reformulate=False),
        gridmix=dict(n_holes=(2, 6), hole_aspect_ratio=1.,
            cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
        manifoldmix=dict(layer=(0, 3)),
        puzzlemix=dict(transport=True, t_batch_size=32, t_size=32,  # t_size for small-scale datasets
            block_num=4, beta=1.2, gamma=0.5, eta=0.2, neigh_size=4, n_labels=3, t_eps=0.8),
=======
        fmix=dict(decay_power=3, size=(32,32), max_soft=0., reformulate=False),
        gridmix=dict(n_holes=(2, 6), hole_aspect_ratio=1.,
            cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
        manifoldmix=dict(layer=(0, 3)),
        puzzlemix=dict(transport=True, t_batch_size=None, t_size=4,  # t_size for small-scale datasets
            block_num=5, beta=1.2, gamma=0.5, eta=0.2, neigh_size=4, n_labels=3, t_eps=0.8),
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
        samix=dict(mask_adjust=0, lam_margin=0.08),  # require pre-trained mixblock
    ),
    backbone=dict(
<<<<<<< HEAD
        type='ResNet_CIFAR',  # CIFAR version
        # type='ResNet_Mix_CIFAR',  # required by 'manifoldmix'
=======
        # type='ResNet_CIFAR',  # CIFAR version
        type='ResNet_Mix_CIFAR',  # required by 'manifoldmix'
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style='pytorch'),
    head=dict(
        type='ClsHead',  # normal CE loss (NOT SUPPORT PuzzleMix, use soft/sigm CE instead)
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=100)
)

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        iter_per_epoch=500,
        save_interval=12500),  # plot every 500 x 25 ep
]

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
<<<<<<< HEAD
runner = dict(type='EpochBasedRunner', max_epochs=200)
=======
runner = dict(type='EpochBasedRunner', max_epochs=400)
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
