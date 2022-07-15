# model settings
model = dict(
    type='MixUpClassification',
    pretrained=None,
    alpha=1,
    mix_mode="mixup",
    mix_args=dict(),
    backbone=dict(
        type='ResNet',
        # type='ResNet_Mix',  # for `ManifoldMix`
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    head=dict(
        type='ClsMixupHead',  # mixup CE loss
        loss=dict(
            type='CrossEntropyLoss', use_soft=False, use_sigmoid=False, loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=2048, num_classes=1000)
)
