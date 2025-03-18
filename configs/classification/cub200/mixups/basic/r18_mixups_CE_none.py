_base_ = [
    '../../../_base_/datasets/cub200/sz224_bs16.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
<<<<<<< HEAD
    pretrained='tools/resnet18-pytorch.pth',
    alpha=0.2,  # float or list
    mix_mode="smoothmix",  # str or list, choose a mixup mode
=======
    pretrained="torchvision://resnet18",
    alpha=1,  # float or list
    mix_mode="mixup",  # str or list, choose a mixup mode
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    mix_args=dict(
        alignmix=dict(eps=0.1, max_iter=100),
        attentivemix=dict(grid_size=32, top_k=None, beta=8),  # AttentiveMix+ in this repo (use pre-trained)
        automix=dict(mask_adjust=0, lam_margin=0),  # require pre-trained mixblock
<<<<<<< HEAD
        adaptivemix=dict(mask_adjust=0, lam_margin=0.03),  # require pre-trained mixblock
=======
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
        fmix=dict(decay_power=3, size=(224,224), max_soft=0., reformulate=False),
        gridmix=dict(n_holes=(2, 6), hole_aspect_ratio=1.,
            cut_area_ratio=(0.5, 1), cut_aspect_ratio=(0.5, 2)),
        manifoldmix=dict(layer=(0, 3)),
        puzzlemix=dict(transport=True, t_batch_size=32, t_size=-1,  # adjust t_batch_size if CUDA out of memory
            mp=None, block_num=4,  # block_num<=4 and mp=2/4 for fast training
            beta=1.2, gamma=0.5, eta=0.2, neigh_size=4, n_labels=3, t_eps=0.8),
        resizemix=dict(scope=(0.1, 0.8), use_alpha=True),
        samix=dict(mask_adjust=0, lam_margin=0.08),  # require pre-trained mixblock
    ),
    backbone=dict(
        type='ResNet',  # normal
        # type='ResNet_Mix',  # required by 'manifoldmix'
        depth=18,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style='pytorch'),
    head=dict(
        type='ClsHead',  # normal CE loss
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=200)
)

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)

# lr scheduler
lr_config = dict(policy='CosineAnnealing', min_lr=0)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
