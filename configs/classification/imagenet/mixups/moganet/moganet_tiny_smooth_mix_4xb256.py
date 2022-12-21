_base_ = [
    '../../../_base_/datasets/imagenet/moga_light_sz224_8xbs128.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=0.2,
    mix_mode="cutmix",
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
        type='MogaNet',
        arch="tiny",
        init_value=1e-5,
        drop_path_rate=0.1,
        stem_norm_cfg=dict(type='BN', eps=1e-5),
        conv_norm_cfg=dict(type='BN', eps=1e-5),
        attn_force_fp32=True,  # force fp32 of gating for fp16 training
    ),
    head=dict(
        type='ClsMixupHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=True,
        in_channels=256, num_classes=1000,
    ),
    init_cfg=[
        dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02, bias=0.),
        dict(type='Constant', layer=['BatchNorm', 'LayerNorm'], val=1., bias=0.)
    ],
)

# data
data = dict(imgs_per_gpu=256, workers_per_gpu=10)

# additional hooks
update_interval = 1  # 256 x 4gpus x 1 accumulates = bs1024
custom_hooks = [
    dict(type='PreciseBNHook',
        num_samples=8192,
        update_all_stats=False,
        interval=1,
    ),
]

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,  # lr = 5e-4 * 1024 / 512 = 1e-3 / bs1024
    weight_decay=0.04, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'layer_scale': dict(weight_decay=0.),
        'scale': dict(weight_decay=0.),
    })

# fp16
use_fp16 = True
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=None, update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
