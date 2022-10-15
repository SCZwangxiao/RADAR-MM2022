import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import REACTORS


@REACTORS.register_module()
class LMF(nn.Module):
    """Low-rank Multimodal Fusion Module.

    Args:
        input_size (int): Number of dimensions in input feature.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 feat_keys,
                 modality_dims,
                 hidden_dims,
                 rank):
        super().__init__()
        assert len(feat_keys) == len(modality_dims)
        self.feat_keys = feat_keys
        self.M = len(modality_dims)
        self.modality_dims = modality_dims
        self.hidden_dims = hidden_dims
        self.rank = rank
        for idx in range(self.M):
            feat_key = feat_keys[idx]
            d_m = modality_dims[idx]
            self.__setattr__(
                f'modality_factor_{feat_key}', 
                nn.Parameter(torch.Tensor(rank, d_m + 1, hidden_dims)))
        self.norm = nn.Sequential(
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU()
        )
    
    def init_weights(self):
        """Initiate the parameters from scratch."""
        for feat_key in self.feat_keys:
            nn.init.uniform_(getattr(self, f'modality_factor_{feat_key}'))

    def forward(self, **kwargs):
        """Defines the computation performed at every call.

        Args:
            imgs (torch.Tensor): The input video feature.
            sents (torch.Tensor): The input video feature.
            audios (torch.Tensor): The input video feature.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        for feat_key, feat in kwargs.items():
            # [N, d_m]
            feat = F.pad(feat, [0, 1], value=1) # 1 pad
            # [N, d_m + 1]
            modality_factor = getattr(self, f'modality_factor_{feat_key}')
            # [r, d_m + 1, d_h]
            kwargs[feat_key] = torch.matmul(feat, modality_factor)
            # [r, N, d_h]
        mm_feats = torch.stack(list(kwargs.values()), dim=3)
        # [r, N, d_h, M]
        mm_feats = torch.prod(mm_feats, dim=3)
        # [r, N, d_h]
        mm_feats = mm_feats.sum(0)
        # [N, d_h]
        return self.norm(mm_feats)
