from math import exp

import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

from ...core import mmit_mean_average_precision, mean_average_precision
from ..builder import build_loss
from ..builder import LINKERS


class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grl_func.apply(x, self.lambda_)


class Discriminator(nn.Module):
    def __init__(self,
                 weight,
                 b,
                 use_grl=False,
                 lambda0=0.05,
                 gamma=10,
                 total_step=2000):
        super().__init__()
        self.weight = weight
        self.b = b
        if use_grl:
            self.grl = GRL(1.0)
            self.lambda0 = lambda0
            self.gamma = gamma
            self.total_step = total_step
        else:
            self.grl = None
    
    def init_weights(self):
        self.p = 0
        self.delta_p = 1 / self.total_step
    
    def step_p(self):
        self.p += self.delta_p
    
    def set_lambda(self):
        self.step_p()
        lambda_ = self.lambda0 * (2 / (1 + exp(-self.gamma * self.p)) - 1)
        self.lambda_ = lambda_
    
    def forward(self, features):
        if self.grl:
            self.set_lambda()
            features = self.grl(features)
        # [N, 2, d]
        features = features @ self.weight
        # [N, 2, 1]
        cls_score = features.squeeze(-1) + self.b
        # [N, 2]
        target = torch.zeros_like(cls_score)
        target[:,1] = 1.0
        # [N, 2]
        loss_d = F.binary_cross_entropy_with_logits(cls_score, target)
        if self.grl:
            loss_d *= self.lambda_
        return loss_d

@LINKERS.register_module()
class OurLinker(nn.Module):
    """Classifier as linker.
    """
    def __init__(self,
                 total_step=2000,
                 loss_cls=dict(type='BCELossWithLogits'),
                 num_layer=1,
                 dropout_ratio=0.0,
                 label_smooth_eps=0,
                 lambda0=0.05,
                 gamma=10):
        super().__init__()
        self.num_layer = num_layer
        self.loss_cls = build_loss(loss_cls)
        self.label_smooth_eps = label_smooth_eps

        self.pos_etype = ('tag', 'HasVideo', 'video')
        self.neg_etype = ('tag', 'NotHasVideo', 'video')
        self.infer_etype = ('tag', 'WhetherHasVideo', 'video')

        self.dropout = nn.Dropout(dropout_ratio)
        self.build_discriminator(num_layer, lambda0, gamma, total_step)

    def init_weights(self):
        nn.init.xavier_normal_(self.D_weight)
        for i in range(self.num_layer):
            self.d_c[i].init_weights()

    def build_discriminator(self, num_layer, lambda0, gamma, total_step):
        self.d_u = nn.ModuleList()
        self.d_c = nn.ModuleList()
        for _ in range(num_layer):
            self.D_weight = nn.Parameter(torch.Tensor(768, 1))
            self.b_weight = nn.Parameter(torch.ones(1))
            self.d_u.append(
                Discriminator(self.D_weight, self.b_weight))
            self.d_c.append(
                Discriminator(self.D_weight, self.b_weight, True, lambda0, gamma, total_step))

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

    def loss_adv(self, mfgs):
        loss_du = []
        loss_dc = []
        for D_u, D_c, mfg in zip(self.d_u, self.d_c, mfgs):
            h_unique = mfg.ndata['h_unique']['tag']
            h_common = mfg.ndata['h_common']['tag']
            loss_du.append(D_u(h_unique))
            loss_dc.append(D_c(h_common))
        loss_du = torch.stack(loss_du).mean() / 2
        loss_dc = torch.stack(loss_dc).mean() / 2
        return loss_du, loss_dc

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
        # Evaluation
        mAP_s = mmit_mean_average_precision(cls_score.detach().cpu().numpy(),
                                          labels.detach().cpu().numpy())
        mAP_l = mean_average_precision(cls_score.detach().cpu().numpy(),
                                          labels.detach().cpu().numpy())
        losses['mAP_sample'] = torch.tensor(
            mAP_s, device=cls_score.device)
        losses['mAP_label'] = torch.tensor(
            mAP_l, device=cls_score.device)
        
        # classification loss
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

        # adversial loss
        loss_du, loss_dc = self.loss_adv(mfgs)
        losses['loss_du'] = loss_du
        losses['loss_dc'] = loss_dc
        return losses