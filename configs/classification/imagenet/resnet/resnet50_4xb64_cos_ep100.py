_base_ = "resnet50_4xb64_step_ep100.py"

# lr scheduler
lr_config = dict(policy='CosineAnnealing', min_lr=0)
