# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm

from ...utils import get_root_logger
from ..builder import BACKBONES


@BACKBONES.register_module()
class Indentity(nn.Module):
    """Indentity backbone.

    Args:
        None
    """

    def __init__(self):
        super().__init__()

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        pass

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input frames feature.
                the size of x is (num_batches, num_clips, hdim).

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """

        return x
