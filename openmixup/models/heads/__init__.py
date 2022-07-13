from .cae_head import CAEHead
from .cls_head import ClsHead
from .cls_mixup_head import ClsMixupHead
from .contrastive_head import ContrastiveHead, HCRHead
from .latent_pred_head import LatentPredictHead, LatentClsHead, LatentCrossCorrelationHead, MoCoV3Head
from .mim_head import MAEPretrainHead, SimMIMHead, MIMHead, MAEFinetuneHead, MAELinprobeHead
from .multi_cls_head import MultiClsHead
from .pmix_block import PixelMixBlock
from .reg_head import RegHead
from .swav_head import MultiPrototypes, SwAVHead
from .vision_transformer_head import VisionTransformerClsHead

__all__ = [
    'CAEHead', 'ClsHead', 'ClsMixupHead', 'ContrastiveHead', 'HCRHead',
    'LatentPredictHead', 'LatentClsHead', 'LatentCrossCorrelationHead',
    'MoCoV3Head', 'MAEPretrainHead', 'MAELinprobeHead', 'MAEFinetuneHead', 'MAELinprobeHead',
    'MultiClsHead', 'MultiPrototypes', 'MIMHead', 'PixelMixBlock', 'RegHead',
    'SwAVHead', 'SimMIMHead', 'VisionTransformerClsHead',
]
