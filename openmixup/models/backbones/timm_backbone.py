# reference: https://github.com/open-mmlab/mmclassification/tree/master/mmcls/models/backbones
# modified from mmclassification timm_backbone.py
try:
    import timm
except ImportError:
    timm = None

from mmcv.cnn.bricks.registry import NORM_LAYERS

from openmixup.utils import get_root_logger
from ..builder import BACKBONES
from .base_backbone import BaseBackbone


def print_timm_feature_info(feature_info):
    """Print feature_info of timm backbone to help development and debug.

    Args:
        feature_info (list[dict] | timm.models.features.FeatureInfo | None):
            feature_info of timm backbone.
    """
    logger = get_root_logger()
    if feature_info is None:
        logger.warning('This backbone does not have feature_info')
    elif isinstance(feature_info, list):
        for feat_idx, each_info in enumerate(feature_info):
            logger.info(f'backbone feature_info[{feat_idx}]: {each_info}')
    else:
        try:
            logger.info(f'backbone out_indices: {feature_info.out_indices}')
            logger.info(f'backbone out_channels: {feature_info.channels()}')
            logger.info(f'backbone out_strides: {feature_info.reduction()}')
        except AttributeError:
            logger.warning('Unexpected format of backbone feature_info')


@BACKBONES.register_module()
class TIMMBackbone(BaseBackbone):
    """Wrapper to use backbones from timm library.

    More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_.
    See especially the document for `feature extraction
    <https://rwightman.github.io/pytorch-image-models/feature_extraction/>`_.

    Args:
        model_name (str): Name of timm model to instantiate.
        in_channels (int): Number of input image channels. Defaults to 3.
        num_classes (int): Number of classes for classification head (used when
            features_only is False). Default to 1000.
        features_only (bool): Whether to extract feature pyramid (multi-scale
            feature maps from the deepest layer at each stride) by using timm
            supported `forward_features()`. Defaults to False.
        pretrained (bool): Whether to load pretrained weights.
            Defaults to False.
        checkpoint_path (str): Path of checkpoint to load at the last of
            ``timm.create_model``. Defaults to empty string, which means
            not loading.
        init_cfg (dict or list[dict], optional): Initialization config dict of
            OpenMMLab projects (removed!). Defaults to None.
        **kwargs: Other timm & model specific arguments.
    """

    def __init__(self,
                 model_name,
                 num_classes=1000,
                 in_channels=3,
                 features_only=False,
                 pretrained=False,
                 checkpoint_path='',
                 **kwargs):
        if timm is None:
            raise RuntimeError(
                'Failed to import timm. Please run "pip install timm". '
                '"pip install dataclasses" may also be needed for Python 3.6.')
        if not isinstance(pretrained, bool):
            raise TypeError('pretrained must be bool, not str for model path')

        super(TIMMBackbone, self).__init__()
        if 'norm_layer' in kwargs:
            kwargs['norm_layer'] = NORM_LAYERS.get(kwargs['norm_layer'])
        self.timm_model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            num_classes=0 if features_only else num_classes,
            **kwargs)
        self.features_only = features_only 

        # reset classifier
        if hasattr(self.timm_model, 'reset_classifier'):
            self.timm_model.reset_classifier(0, '')

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True

        feature_info = getattr(self.timm_model, 'feature_info', None)
        print_timm_feature_info(feature_info)

    def forward(self, x):
        if self.features_only:
            features = self.timm_model.forward_features(x)
        else:
            features = self.timm_model(x)
        if isinstance(features, (list, tuple)):
            features = list(features)
        else:
            features = [features]
        return features
