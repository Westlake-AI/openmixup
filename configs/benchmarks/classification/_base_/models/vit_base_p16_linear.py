# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='MIMVisionTransformer',
        arch='base',
        img_size=224,
        patch_size=16,
        final_norm=True,
        finetune=False),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=False, multi_label=False,  # no gap in ViT
        in_channels=768, num_classes=1000),
)
