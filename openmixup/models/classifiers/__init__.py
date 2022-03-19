from .base_model import BaseModel
from .classification import Classification
from .mixup_classification import MixUpClassification
from .automix_V1plus import AutoMixup
from .automix_V2 import AutoMixup_V2


__all__ = [
    'BaseModel', 'Classification',
    'MixUpClassification', 'AutoMixup', 'AutoMixup_V2',
]
