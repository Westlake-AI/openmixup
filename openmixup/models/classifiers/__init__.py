from .classification import Classification
from .mixup_classification import MixUpClassification
from .automix_V1plus import AutoMixup
from .automix_V2 import AutoMixup_V2
from .representation import Representation


__all__ = [
    'Classification', 'Representation',
    'MixUpClassification', 'AutoMixup', 'AutoMixup_V2',
]
