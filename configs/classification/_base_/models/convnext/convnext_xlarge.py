# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=[0.8, 1.0,],
    mix_mode=["mixup", "cutmix",],
    mix_args=dict(),
    backbone=dict(
        type='ConvNeXt',
        arch='xlarge',
        out_indices=(3,),  # x-1: stage-x
        act_cfg=dict(type='GELU'),
        drop_path_rate=0.5,
        gap_before_final_norm=True,
    ),
    head=dict(
        type='ClsMixupHead',
        loss=dict(type='LabelSmoothLoss',
            label_smooth_val=0.1, num_classes=1000, mode='original', loss_weight=1.0),
        with_avg_pool=False,
        in_channels=2048, num_classes=1000)
)
