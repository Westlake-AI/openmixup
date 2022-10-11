from .a2mim import A2MIM
from .barlowtwins import BarlowTwins
from .beit import BEiT
from .byol import BYOL
from .cae import CAE
from .deepcluster import DeepCluster
from .densecl import DenseCL
from .mae import MAE
from .maskfeat import MaskFeat
from .moco import MOCO
from .moco_mix import MoCoMix
from .moco_samix import MoCoSAMix
from .mocov3 import MoCoV3
from .npid import NPID
from .odc import ODC
from .rotation_pred import RotationPred
from .relative_loc import RelativeLoc
from .simclr import SimCLR
from .simclr_mix import SimCLRMix
from .simmim import SimMIM
from .simsiam import SimSiam
from .swav import SwAV

__all__ = [
    'A2MIM', 'BarlowTwins', 'BEiT', 'BYOL', 'CAE', 'DeepCluster', 'DenseCL',
    'MAE', 'MaskFeat', 'MOCO', 'MoCoMix', 'MoCoSAMix', 'MoCoV3',
    'NPID', 'ODC', 'RelativeLoc', 'RotationPred', 'SimCLR', 'SimCLRMix',
    'SimMIM', 'SimSiam', 'SwAV',
]
