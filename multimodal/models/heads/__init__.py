from .base import BaseHead
from .cma_head import CMAHead
from .mall_head import MALLHead
from .mlgcn_head import MLGCNHead
from .vanilla_multimodal_head import VanillaMultiModalHead
from .tsn_head import TSNHead
from .nextvlad_head import NextVLADHead

__all__ = [
    'TSNHead', 'BaseHead', 'MLGCNHead', 'CMAHead', 'MALLHead',
    'VanillaMultiModalHead', 'NextVLADHead'
]
