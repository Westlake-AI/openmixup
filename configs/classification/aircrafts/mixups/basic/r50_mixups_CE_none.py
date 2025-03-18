_base_ = "r18_mixups_CE_none.py"

# model settings
model = dict(
<<<<<<< HEAD
=======
    pretrained="torchvision://resnet50",
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    alpha=1,  # float or list
    mix_mode="mixup",  # str or list, choose a mixup mode
    backbone=dict(
        type='ResNet',  # normal
        # type='ResNet_Mix',  # required by 'manifoldmix'
        depth=50,
        num_stages=4,
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style='pytorch'),
    head=dict(
        type='ClsHead',  # normal CE loss
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=2048, num_classes=100)
)
