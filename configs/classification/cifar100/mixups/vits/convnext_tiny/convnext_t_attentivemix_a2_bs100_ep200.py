_base_ = "../convnext_t_mixups_bs100.py"

# model settings
model = dict(
    pretrained=None,
    pretrained_k="torchvision://resnet50",
    alpha=2,  # float or list
    mix_mode="attentivemix",
    backbone_k=dict(  # PyTorch pre-trained R-18 is required for attentivemix+
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
