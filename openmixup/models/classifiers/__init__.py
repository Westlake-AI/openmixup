from .base_model import BaseModel
from .classification import Classification
from .mixup_classification import MixUpClassification
from .mim_classification import MIMClassification
from .automix_V1plus import AutoMixup

__all__ = [
    'BaseModel', 'Classification', 'MixUpClassification', 'MIMClassification',
    'AutoMixup',
]
