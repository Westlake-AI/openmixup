from .cls_head import ClsHead
from .cls_mixup_head import ClsMixupHead
from .contrastive_head import ContrastiveHead
from .latent_pred_head import LatentPredictHead, LatentClsHead, MoCoV3Head
from .mae_head import MAEPretrainHead
from .multi_cls_head import MultiClsHead
from .pmix_block import PixelMixBlock
from .swav_head import MultiPrototypes, SwAVHead
from .vision_transformer_head import VisionTransformerClsHead


__all__ = [
    'ContrastiveHead', 'ClsHead', 'ClsMixupHead', 'LatentPredictHead', 'LatentClsHead',
    'MoCoV3Head', 'MAEPretrainHead', 'MultiClsHead', 'MultiPrototypes', 'SwAVHead',
    'VisionTransformerClsHead',
    'PixelMixBlock',
]
