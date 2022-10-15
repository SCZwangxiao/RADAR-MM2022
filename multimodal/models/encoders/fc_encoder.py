import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import ENCODERS


@ENCODERS.register_module()
class FcEncoder(nn.Module):
    """Fully-connected layer encoder.

    Args:
        input_dims (int): Number of dimensions in input feature.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 input_dims,
                 hidden_dims,
                 dropout_ratio=0.4,
                 init_std=0.01,
                 **kwargs):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc1 = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU()
        )

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc1, std=self.init_std)

    def forward(self, x, mask=None, length=None):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            mask  (torch.Tensor): The mask of input sequence.
            length (int): length of each sequence.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, input_dims] / [N, length, input_dims]
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc1(x)
        # [N, hidden_dims]
        return x
