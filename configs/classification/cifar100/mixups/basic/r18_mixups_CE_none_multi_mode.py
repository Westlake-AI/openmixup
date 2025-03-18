_base_ = "r18_mixups_CE_none.py"

# model settings
model = dict(
    alpha=[0.1, 1, 1,],  # list of alpha
    mix_mode=["mixup", "cutmix", "vanilla",],  # list of chosen mixup modes
    mix_prob=None,  # list of applying probs (sum=1), None for random applying
    mix_repeat=1,  # times of repeating mixup aug
    mix_args=dict(
        alignmix=dict(eps=0.1, max_iter=100),
        attentivemix=dict(grid_size=32, top_k=None, beta=8),  # AttentiveMix+ in this repo (use pre-trained)
        automix=dict(mask_adjust=0, lam_margin=0),  # require pre-trained mixblock
        fmix=dict(decay_power=3, size=(32,32), max_soft=0., reformulate=False),
        gridmix=dict(n_holes=(2, 6), hole_aspect_ratio=1.,
            cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
        manifoldmix=dict(layer=(0, 3)),
        puzzlemix=dict(transport=True, t_batch_size=None, t_size=4,  # t_size for small-scale datasets
            block_num=5, beta=1.2, gamma=0.5, eta=0.2, neigh_size=4, n_labels=3, t_eps=0.8),
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
        samix=dict(mask_adjust=0, lam_margin=0.08),  # require pre-trained mixblock
    ),
<<<<<<< HEAD
    backbone=dict(
        type='ResNet_CIFAR',  # CIFAR version
        # type='ResNet_Mix_CIFAR',  # required by 'manifoldmix'
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style='pytorch'),
    head=dict(
        type='ClsHead',  # normal CE loss (NOT SUPPORT PuzzleMix, use soft/sigm CE instead)
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=100)
=======
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
)

# additional hooks
custom_hooks = [
    dict(type='SAVEHook',
        iter_per_epoch=500,
<<<<<<< HEAD
        save_interval=12500),  # plot every 500 x 25 ep
]

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
=======
        save_interval=12500,  # plot every 500 x 25 ep
    )
]
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
