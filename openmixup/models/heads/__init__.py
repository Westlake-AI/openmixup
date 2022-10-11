from .cls_head import ClsHead
from .cls_mixup_head import ClsMixupHead
from .contrastive_head import ContrastiveHead, HCRHead
from .latent_pred_head import LatentPredictHead, LatentClsHead, LatentCrossCorrelationHead, MoCoV3Head
from .mim_head import A2MIMHead, MAEPretrainHead, MAEFinetuneHead, MAELinprobeHead, SimMIMHead
from .multi_cls_head import MultiClsHead
from .norm_linear_head import NormLinearClsHead
from .pmix_block import PixelMixBlock
from .reg_head import RegHead
from .swav_head import MultiPrototypes, SwAVHead
from .tokenizer_head import BEiTHead, CAEHead
from .vision_transformer_head import VisionTransformerClsHead

__all__ = [
    'A2MIMHead', 'BEiTHead', 'CAEHead', 'ClsHead', 'ClsMixupHead', 'ContrastiveHead', 'HCRHead',
    'LatentPredictHead', 'LatentClsHead', 'LatentCrossCorrelationHead',
    'MoCoV3Head', 'MAEPretrainHead', 'MAELinprobeHead', 'MAEFinetuneHead', 'MAELinprobeHead',
    'MultiClsHead', 'MultiPrototypes', 'NormLinearClsHead', 'PixelMixBlock', 'RegHead',
    'SwAVHead', 'SimMIMHead', 'VisionTransformerClsHead',
]
