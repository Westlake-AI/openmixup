from .conv_necks import ConvNeck
from .hr_necks import HRFuseScales
from .mim_neck import BEiTNeck, CAENeck, MAEPretrainDecoder, NonLinearMIMNeck, SimMIMNeck
from .mlp_necks import (AvgPoolNeck, MaskPoolNeck, GeneralizedMeanPooling, LinearNeck,
<<<<<<< HEAD
                        RelativeLocNeck, ODCNeck, MoCoV2Neck, NonLinearNeck, SwAVNeck, DenseCLNeck)
from .transformer_necks import TransformerNeck

__all__ = [
    'AvgPoolNeck', 'BEiTNeck', 'CAENeck', 'ConvNeck', 'HRFuseScales', 'DenseCLNeck',
=======
                        RelativeLocNeck, ODCNeck, MoCoV2Neck, NonLinearNeck, SwAVNeck,
                        DenseCLNeck, DINONeck)
from .transformer_necks import TransformerNeck

__all__ = [
    'AvgPoolNeck', 'BEiTNeck', 'CAENeck', 'ConvNeck', 'HRFuseScales', 'DenseCLNeck', 'DINONeck',
>>>>>>> db2c4ac (update some vit-based mixup methods and fix robustness eval tasks)
    'GeneralizedMeanPooling', 'LinearNeck', 'MoCoV2Neck', 'NonLinearNeck', 'ODCNeck', 'RelativeLocNeck',
    'MAEPretrainDecoder', 'MaskPoolNeck', 'NonLinearMIMNeck', 'SimMIMNeck', 'SwAVNeck',
    'TransformerNeck',
]
