from .byol import BYOL
from .deepcluster import DeepCluster
from .moco import MOCO
from .moco_mix import MOCO_Mix
from .moco_automix_v2 import MOCO_AutoMix_V2
from .npid import NPID
from .odc import ODC
from .rotation_pred import RotationPred
from .relative_loc import RelativeLoc
from .simclr import SimCLR
from .simclr_mix import SimCLR_Mix


__all__ = [
    'BYOL', 'DeepCluster', 'MOCO', 'MOCO_Mix', 'MOCO_AutoMix_V2',
    'NPID', 'ODC', 'RelativeLoc', 'RotationPred', 'SimCLR', 'SimCLR_Mix',
]
