# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import trunc_normal_init
import dgl
from dgl.nn.pytorch.conv import GATConv
import numpy as np

from ...core import mmit_mean_average_precision, mean_average_precision
from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class CMAHead(BaseHead):
    """CMA classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 tag_embedding_path,
                 graph_path,
                 label_map_path,
                 num_layers=1,
                 num_heads=4,
                 tag_emb_dim=768,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout_ratio=0.4,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        # Build label gnn
        tag2idx, tag_graph, tag_feats = self.generate_label_graph(
            tag_embedding_path, graph_path, label_map_path)
        self.tag2idx = tag2idx
        self.tag_graph = tag_graph
        self.register_buffer('tag_feats', tag_feats)
        self.tag_feats_id = nn.Parameter(torch.zeros_like(tag_feats))
        self.tag_fc = nn.Sequential(
            nn.Linear(tag_emb_dim, in_channels),
            nn.LayerNorm(tag_emb_dim),
            nn.GELU())
        self.tag_gnns = nn.ModuleList()
        self.gnn_norm = nn.ModuleList()
        for _ in range(num_layers):
            gnn = GATConv(
                in_feats=in_channels,
                out_feats=in_channels // num_heads,
                num_heads=num_heads,
                feat_drop=dropout_ratio,
                attn_drop=0.2,
                residual=True,
                activation=F.gelu)
            self.tag_gnns.append(gnn)
            self.gnn_norm.append(nn.LayerNorm(in_channels))
        # video encoder
        self.video_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU())
        # classification head
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.w = nn.Parameter(torch.empty(1, num_classes, in_channels))
        self.b =  nn.Parameter(torch.zeros(1, num_classes))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        nn.init.xavier_normal_(self.tag_feats_id)
        nn.init.xavier_normal_(self.w)

    @staticmethod
    def generate_label_graph(embedding_path, graph_path, label_map_path):
        # Load tag index
        with open(label_map_path, 'r') as F:
            lines = F.readlines()
        tag2idx = {}
        idx2tag = {}
        for tag in lines:
            tag = tag.strip()
            tag2idx[tag] = len(tag2idx)
            idx2tag[len(idx2tag)] = tag
        # Load edges
        sources = []
        targets = []
        with open(graph_path, 'r') as F:
            graph_edges = json.load(F)
        for child, parents in graph_edges.items():
            child_id = tag2idx[child]
            for parent in parents:
                parent_id = tag2idx[parent]
                sources.append(child_id)
                targets.append(parent_id)
        tag_graph = dgl.graph((sources, targets))
        tag_graph = dgl.add_self_loop(tag_graph)
        # Load features
        tag_feats = []
        for tag_nid in range(len(idx2tag)):
            tag = idx2tag[tag_nid]
            feat_filename = osp.join(embedding_path, tag) + '.npy'
            feat = np.load(feat_filename)
            feat = torch.tensor(feat)
            tag_feats.append(feat)
        tag_feats = torch.stack(tag_feats)
        return tag2idx, tag_graph, tag_feats

    def video_encoding(self, x, num_segs):
        # [N, num_segs, in_channels]
        x = self.video_fc(x)
        # [N, num_segs, in_channels]
        return x

    def tag_encoding(self, tag_feats, tag_feats_id, tag_graph):
        tag_graph = tag_graph.to(tag_feats.device)
        tag_feats = self.tag_fc(tag_feats + tag_feats_id)
        # [num_classes, in_channels]
        for gnn, norm in zip(self.tag_gnns, self.gnn_norm):
            tag_feats = gnn(tag_graph, tag_feats)
            # [num_classes, num_heads, in_channels//num_heads]
            tag_feats = tag_feats.reshape(self.num_classes, -1)
            tag_feats = norm(tag_feats)
        # [num_classes, in_channels]
        return tag_feats

    def predictor(self, x, tag_feats):
        """Return classification logits"""
        B, num_segs, h = x.shape
        tag_feats = tag_feats.expand(B, self.num_classes, h)
        # [B, num_classes, h]
        z = torch.einsum('bnk,btk->bnt', tag_feats, x)
        z = F.relu(z)
        z = F.softmax(z, dim=2)
        # [B, num_classes, num_segs]
        h = torch.bmm(z, x)
        # [B, num_classes, h]
        if self.dropout is not None:
            h = self.dropout(h)
        # [B, num_classes, h]
        cls_score = (h * self.w).sum(2) + self.b
        # [B, num_classes]
        return cls_score

    def forward(self, x, num_segs):
        x = self.video_encoding(x, num_segs)
        # [B, num_segs, in_channels]
        tag_feats = self.tag_encoding(
            self.tag_feats, self.tag_feats_id, self.tag_graph)
        # [num_classes, in_channels]
        cls_score = self.predictor(x, tag_feats)
        # [B, num_classes]
        return cls_score
    
    def build_labels(self, cls_score, sparse_labels):
        batch_size = len(sparse_labels)
        labels = torch.zeros_like(cls_score)
        for b, sparse_label in enumerate(sparse_labels):
            labels[b, sparse_label] = 1.
        return labels

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        
        labels = self.build_labels(cls_score, labels)
        
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class:
            cls_score_cpu = cls_score.detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()
            mAP_sample = mmit_mean_average_precision(cls_score_cpu, labels_cpu)
            losses['mAP_sample'] = torch.tensor(mAP_sample, device=cls_score.device)
            mAP_label = mean_average_precision(cls_score_cpu, labels_cpu)
            losses['mAP_label'] = torch.tensor(mAP_label, device=cls_score.device)
            if self.label_smooth_eps != 0:
                labels = ((1 - self.label_smooth_eps) * labels +
                          self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses