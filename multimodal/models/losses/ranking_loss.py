import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss

@LOSSES.register_module()
class LinkPredMarginLoss(BaseWeightedLoss):
    def __init__(self, margin=1.0, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)
        self.margin = margin

    def _forward(self, pos_score, neg_score):
        n_edges = pos_score.shape[0]
        return (self.margin - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()

@LOSSES.register_module()
class LinkPredBPRLoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

    def _forward(self, pos_score, neg_score):
        n_edges = pos_score.shape[0]
        neg_score = neg_score.view(n_edges, -1)
        return - torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

@LOSSES.register_module()
class LinkPredInfoNCELoss(BaseWeightedLoss):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)
        self.reduction = reduction

    def _forward(self, pos_score, neg_score):
        n_edges = pos_score.shape[0]
        pos_score = pos_score.view(-1, 1)
        neg_score = neg_score.view(n_edges, -1)
        # First index in last dimension are the positive samples
        logits = torch.cat([pos_score, neg_score], dim=1)
        labels = torch.zeros(n_edges, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels, reduction=self.reduction)