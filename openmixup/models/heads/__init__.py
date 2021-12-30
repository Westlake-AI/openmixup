from .cls_head import ClsHead
from .cls_mixup_head import ClsMixupHead
from .contrastive_head import ContrastiveHead
from .latent_pred_head import LatentPredictHead
from .multi_cls_head import MultiClsHead
from .pmix_block_V2 import PixelMixBlock_V2


__all__ = [
    'ContrastiveHead', 'ClsHead', 'ClsMixupHead', 'LatentPredictHead', 'MultiClsHead',
    'PixelMixBlock_V2',
]
