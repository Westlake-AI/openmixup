from .fmix import fmix
from .mixup_input import cutmix, mixup, resizemix, saliencymix, smoothmix
from .mixup_saliency import attentivemix, puzzlemix

__all__ = [
    'cutmix', 'fmix', 'mixup', 'resizemix', 'saliencymix', 'smoothmix',
    'attentivemix', 'puzzlemix',
]
