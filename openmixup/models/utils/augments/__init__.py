from .fmix import fmix
from .mixup_input import cutmix, mixup, saliencymix, resizemix
from .mixup_saliency import attentivemix, puzzlemix


__all__ = (
    'cutmix', 'mixup', 'saliencymix', 'resizemix', 'fmix', 'attentivemix', 'puzzlemix',
)