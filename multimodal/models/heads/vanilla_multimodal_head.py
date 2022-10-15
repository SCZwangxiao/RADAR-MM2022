import torch.nn as nn
from mmcv.cnn import trunc_normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class VanillaMultiModalHead(BaseHead):
    """Vanilla classification head for MultiModalHead recognizer.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout_ratio=0.4,
                 init_std=0.02,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        # [N, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
