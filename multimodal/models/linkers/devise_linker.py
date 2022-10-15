import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

from ...core import mmit_mean_average_precision, mean_average_precision
from ..builder import build_loss
from ..builder import LINKERS


@LINKERS.register_module()
class DeviseLinker(nn.Module):
    """Classifier as linker.
    """
    def __init__(self,
                 loss_cls=dict(type='BCELossWithLogits'),
                 dropout_ratio=0.0,
                 label_smooth_eps=0):
        super().__init__()
        self.loss_cls = build_loss(loss_cls)
        self.label_smooth_eps = label_smooth_eps

        self.pos_etype = ('tag', 'HasVideo', 'video')
        self.neg_etype = ('tag', 'NotHasVideo', 'video')
        self.infer_etype = ('tag', 'WhetherHasVideo', 'video')

        self.dropout = nn.Dropout(dropout_ratio)

    def init_weights(self):
        pass
    
    def hetero_dot_product_predictor(self, graph, etype):
        with graph.local_scope():
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score'].squeeze(-1)

    def forward(self, input_nodes, pair_graph, mfgs):
        """
        Args:
            input_nodes
            output_nodes
            mfgs
        Returns:
            cls_score
            labels
        """
        pair_graph.ndata['h'] = {
            'video': self.dropout(pair_graph.ndata['h']['video']),
            'tag': self.dropout(pair_graph.ndata['h']['tag'])}
        labels = pair_graph.adj(transpose=True, ctx=pair_graph.device, etype=self.pos_etype).to_dense()
        # [B, num_labels]
        pos_score = self.hetero_dot_product_predictor(pair_graph, self.pos_etype)
        neg_score = self.hetero_dot_product_predictor(pair_graph, self.neg_etype)
        pos_links = pair_graph.adj(ctx=pair_graph.device, etype=self.pos_etype)._indices()
        neg_links = pair_graph.adj(ctx=pair_graph.device, etype=self.neg_etype)._indices()
        links = torch.cat([pos_links, neg_links], dim=1)
        score = torch.cat([pos_score, neg_score])
        cls_score = torch.sparse_coo_tensor(links, score).t().to_dense()
        # [B, num_labels]
        return cls_score, labels
    
    def infer(self, input_nodes, pair_graph, mfgs):
        num_tags = pair_graph.num_nodes('tag')
        cls_score = self.hetero_dot_product_predictor(pair_graph, etype=self.infer_etype)
        # [num_labels*B]
        cls_score = cls_score.reshape(-1, num_tags)
        # [B, num_labels]
        return cls_score
    
    def loss(self, cls_score, labels, pair_graph, mfgs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'mAP_sample', 'mAP_label'(optional).
        """
        losses = dict()

        mAP_s = mmit_mean_average_precision(cls_score.detach().cpu().numpy(),
                                          labels.detach().cpu().numpy())
        mAP_l = mean_average_precision(cls_score.detach().cpu().numpy(),
                                          labels.detach().cpu().numpy())
        losses['mAP_sample'] = torch.tensor(
            mAP_s, device=cls_score.device)
        losses['mAP_label'] = torch.tensor(
            mAP_l, device=cls_score.device)
        
        if self.label_smooth_eps != 0:
            num_tags = labels.size(1)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / num_tags)
        loss_cls = self.loss_cls(cls_score, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses