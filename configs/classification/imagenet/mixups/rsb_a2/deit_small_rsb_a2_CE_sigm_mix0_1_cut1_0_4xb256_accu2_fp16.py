_base_ = [
    '../../../_base_/datasets/imagenet/rsb_a2_sz224_8xbs256.py',
    '../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.1, 1.0,],  # RSB setting
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(
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
        type='VisionTransformer',
        arch='deit-small',
        img_size=224, patch_size=16,
    ),
    head=dict(
        type='VisionTransformerClsHead',  # mixup CE + label smooth
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        in_channels=384, num_classes=1000)
)

# interval for accumulate gradient
update_interval = 2  # 256 x 4gpus x 2 accumulates = bs2048

# optimizer
optimizer = dict(type='LAMB', lr=0.005, weight_decay=0.02,  # RSB A2
                 paramwise_options={
                    '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
                    'norm': dict(weight_decay=0.),
                    'bias': dict(weight_decay=0.),
                    'cls_token': dict(weight_decay=0.),
                    'pos_embed': dict(weight_decay=0.),
                })
# apex
use_fp16 = True
fp16 = dict(type='apex', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# lr scheduler
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,  # warmup 5 epochs.
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
