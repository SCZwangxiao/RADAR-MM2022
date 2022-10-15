import torch
from torch import nn
import dgl

from ...core import mmit_mean_average_precision, mean_average_precision
from ..builder import build_loss
from ..builder import LINKERS


@LINKERS.register_module()
class ClassifierLinker(nn.Module):
    """Classifier as linker.
    """
    def __init__(self, 
                 num_tags,
                 feat_dim,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout_ratio=0,
                 label_smooth_eps=0.0):
        super().__init__()
        self.tag_embed = nn.Embedding(
            num_tags,
            feat_dim)
        self.tag_b = nn.Embedding(
            num_tags,
            1)
        self.loss_cls = build_loss(loss_cls)
        self.label_smooth_eps = label_smooth_eps

        if dropout_ratio != 0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.pos_etype = ('tag', 'HasVideo', 'video')

    def init_weights(self):
        pass

    def forward(self, input_nodes, pair_graph, mfgs):
        """
        Args:
            imgs (B, d)
            tag_embedding (num_tags, d)
        Returns:
            cls_score (B, num_tags)
        """
        labels = pair_graph.adj(transpose=True, ctx=pair_graph.device, etype=self.pos_etype).to_dense()
        # [B, num_tags]
        imgs = mfgs[0].srcdata['feat']['video']
        if self.dropout is not None:
            imgs = self.dropout(imgs)
        # [B, feat_dim]
        tag_ids = mfgs[0].srcdata[dgl.NID]['tag']
        # [B, num_tags]
        tag_embedding = self.tag_embed(tag_ids)
        # [B, feat_dim]
        b = self.tag_b(tag_ids).t()
        # [1, num_tags]
        cls_score = imgs @ tag_embedding.t() + b
        # [B, num_tags]
        return cls_score, labels
    
    def infer(self, input_nodes, pair_graph, mfgs):
        imgs = mfgs[0].srcdata['feat']['video']
        if self.dropout is not None:
            imgs = self.dropout(imgs)
        # [B, feat_dim]
        tag_ids = mfgs[0].srcdata[dgl.NID]['tag']
        # [B, num_tags]
        tag_embedding = self.tag_embed(tag_ids)
        # [B, feat_dim]
        b = self.tag_b(tag_ids).t()
        # [1, num_tags]
        cls_score = imgs @ tag_embedding.t() + b
        # [B, num_tags]
        return cls_score
    
    def loss(self, cls_score, labels, pair_graph, mfgs, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'mAP_sample', 'mAP_label'(optional).
        """
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_tags \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        mAP_s = mmit_mean_average_precision(cls_score.detach().cpu().numpy(),
                                          labels.detach().cpu().numpy())
        mAP_l = mean_average_precision(cls_score.detach().cpu().numpy(),
                                          labels.detach().cpu().numpy())
        losses['mAP_sample'] = torch.tensor(
            mAP_s, device=cls_score.device)
        losses['mAP_label'] = torch.tensor(
            mAP_l, device=cls_score.device)
        if self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_tags)

        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses