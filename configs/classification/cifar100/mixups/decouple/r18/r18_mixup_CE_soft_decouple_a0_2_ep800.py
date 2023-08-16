_base_ = '../r18_mixups_CE_soft_decouple.py'

# model settings
model = dict(
    alpha=0.2,
    mix_mode="mixup",
    head=dict(
        type='ClsMixupHead',  # soft CE decoupled mixup
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0,
            use_soft=True, use_sigmoid=False, use_mix_decouple=True,  # decouple mixup CE
        ),
        with_avg_pool=True, multi_label=True, two_hot=False, two_hot_scale=1,  # not two-hot
        lam_scale_mode='pow', lam_thr=1, lam_idx=0.5,  # lam rescale, default as linear
        eta_weight=dict(eta=0.01, mode="more", thr=0.5),
        in_channels=512, num_classes=100)
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=800)
