import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import REACTORS


@REACTORS.register_module()
class BasicReactor(nn.Module):
    """Basic Reactor for multimodal fusion.

    Args:
        input_size (int): Number of dimensions in input feature.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 feat_keys=None,
                 operations=['concat']):
        super().__init__()
        self.operations = operations

    def forward(self, **kwargs):
        """Defines the computation performed at every call.

        Args:
            imgs (torch.Tensor): The input video feature.
            sents (torch.Tensor): The input video feature.
            audios (torch.Tensor): The input video feature.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, input_size]
        mm = []
        feats = list(kwargs.values())
        for operation in self.operations:
            if operation == 'concat':
                for feat in feats:
                    mm.append(feat)
            elif operation == 'add':
                feat = sum(feats)
                mm.append(feat)
            elif operation == 'multiply':
                feat = feats[0]
                for f in feats[1:]:
                    feat *= f
                mm.append(feat)
        mm = torch.cat(mm, dim=1)
        return mm

    def init_weights(self):
        pass