from .adaptive_mask import AdaptiveMask
from .cls_head import ClsHead
from .cls_mixup_head import ClsMixupHead, ClsUncertainMixupHead
from .cls_mlp_head import (EfficientFormerClsHead, MetaFormerClsHead, LeViTClsHead, StackedLinearClsHead,
                           VanillaNetClsHead)
from .contrastive_head import ContrastiveHead, HCRHead
from .dino_head import DINOHead
from .latent_pred_head import LatentPredictHead, LatentClsHead, LatentCrossCorrelationHead, MoCoV3Head
from .mim_head import A2MIMHead, MAEPretrainHead, MAEFinetuneHead, MAELinprobeHead, SimMIMHead
from .multi_cls_head import MultiClsHead
from .norm_linear_head import NormLinearClsHead
from .pmix_block import PixelMixBlock
from .reg_head import RegHead
from .swav_head import MultiPrototypes, SwAVHead
from .tokenizer_head import BEiTHead, CAEHead
from .vision_transformer_head import VisionTransformerClsHead, DistillationVisionTransformerClsHead

__all__ = [
    'A2MIMHead', 'AdaptiveMask', 'BEiTHead', 'CAEHead', 'ClsHead', 'ClsMixupHead', 'ClsUncertainMixupHead'
    'ContrastiveHead', 'DINOHead',
    'EfficientFormerClsHead', 'HCRHead', 'MetaFormerClsHead', 'LeViTClsHead', 'StackedLinearClsHead', 'VanillaNetClsHead',
    'LatentPredictHead', 'LatentClsHead', 'LatentCrossCorrelationHead',
    'MoCoV3Head', 'MAEPretrainHead', 'MAELinprobeHead', 'MAEFinetuneHead', 'MAELinprobeHead',
    'MultiClsHead', 'MultiPrototypes', 'NormLinearClsHead', 'PixelMixBlock', 'RegHead',
    'SwAVHead', 'SimMIMHead', 'VisionTransformerClsHead', 'DistillationVisionTransformerClsHead',
]
