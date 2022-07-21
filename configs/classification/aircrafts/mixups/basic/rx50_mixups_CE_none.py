_base_ = "r18_mixups_CE_none.py"

# model settings
model = dict(
    alpha=1,  # float or list
    mix_mode="mixup",  # str or list, choose a mixup mode
    backbone=dict(
        type='ResNeXt',  # normal
        # type='ResNeXt_Mix',  # required by 'manifoldmix'
        depth=50,
        groups=32, width_per_group=4,  # 32x4d
        out_indices=(3,),  # no conv-1, x-1: stage-x
        style='pytorch'),
    head=dict(
        type='ClsHead',  # normal CE loss
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=2048, num_classes=100)
)
