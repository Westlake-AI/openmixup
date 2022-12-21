_base_ = [
    '../../../_base_/datasets/imagenet/rsb_a3_sz160_4xbs512.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=0.1,  # str of list
    mix_mode="mixup",  # str or list, choose a mixup mode
    mix_args=dict(
        alignmix=dict(eps=0.1, max_iter=100),
        attentivemix=dict(grid_size=32, top_k=None, beta=8),  # AttentiveMix+ in this repo (use pre-trained)
        automix=dict(mask_adjust=0, lam_margin=0),  # require pre-trained mixblock
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
        # type='ResNet',  # normal
        type='ResNet_Mix',  # required by 'manifoldmix'
        depth=50,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        norm_cfg=dict(type='SyncBN'),
        style='pytorch'),
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='CrossEntropyLoss',  # default CE loss
            use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, two_hot=False,
        in_channels=2048, num_classes=1000)
)

# data
data = dict(imgs_per_gpu=512, workers_per_gpu=10)

# interval for accumulate gradient
update_interval = 1  # 512 x 4gpus x 1 accumulates = bs2048

# optimizer
optimizer = dict(type='LAMB', lr=0.008, weight_decay=0.02,
                 paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
                    'bias': dict(weight_decay=0.)})
# apex
use_fp16 = True
fp16 = dict(type='apex', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=None, update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1.0e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
