# model settings
model = dict(
    type='DINO',
    base_momentum=0.99,
    backbone=dict(
        type='VisionTransformer',
        arch='base',
        img_size=224,
        patch_size=16,
    ),
    neck=dict(
        type='DINONeck',
        in_channels=768,
        out_channels=65536,
        hidden_channels=2048,
        bottleneck_channels=256),
    head=dict(
        type='DINOHead',
        out_channels=65536,
        num_crops=10,
        student_temp=0.1,
        center_momentum=0.9)
)
