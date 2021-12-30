from .classification import Classification
from .mixup_classification import MixUpClassification
from .mixup_momentum_V1plus import AutoMixup_V1plus
from .mixup_momentum_V2 import AutoMixup_V2
from .representation import Representation


__all__ = [
    'Classification', 'Representation',
    'MixUpClassification', 'AutoMixup_V1plus', 'AutoMixup_V2',
]
