_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(frozen_stages=-1, norm_cfg=norm_cfg, norm_eval=False),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(type='Shared4Conv1FCBBoxHead', norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

evaluation = dict(save_best='auto')
checkpoint_config = dict(max_keep_ckpts=1)
