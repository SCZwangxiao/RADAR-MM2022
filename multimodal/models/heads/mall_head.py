# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import json
import random

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
class MALLHead(BaseHead):
    """MALL-CNN classification head.

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
                 graph_path,
                 label_map_path,
                 num_layers=1,
                 cluster_size=32,
                 max_neighbors=4,
                 num_heads=4,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout_ratio=0.4,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.num_layers = num_layers
        self.cluster_size = cluster_size
        self.max_neighbors = max_neighbors
        self.dropout_ratio = dropout_ratio
        # Build label gnn
        tag2idx, graph_edges = self.generate_label_graph(graph_path, label_map_path)
        self.tag2idx = tag2idx
        self.graph_edges = graph_edges
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
        ## Multiple Instance Video Learning
        self.mil_P = nn.Linear(in_channels, in_channels*cluster_size, bias=False)
        self.mil_Q = nn.Linear(in_channels, in_channels*cluster_size, bias=False)
        self.mil_a = nn.Parameter(torch.empty(cluster_size, in_channels))
        self.mil_dropout = nn.Dropout(dropout_ratio)
        self.mil_proj = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(cluster_size, num_classes, 1))
        ## Label Attention Transformer
        self.lat_conv = nn.Sequential(
            nn.Conv1d(num_classes, 1, 1),
            nn.Dropout(dropout_ratio))
        # classification head
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.pred = nn.Linear(in_channels, 1)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        nn.init.xavier_normal_(self.mil_a)

    @staticmethod
    def generate_label_graph(graph_path, label_map_path):
        # Load tag index
        with open(label_map_path, 'r') as F:
            lines = F.readlines()
        tag2idx = {}
        for tag in lines:
            tag = tag.strip()
            tag2idx[tag] = len(tag2idx)
        # Load edges
        with open(graph_path, 'r') as F:
            graph_edges = json.load(F)
        return tag2idx, graph_edges

    @staticmethod
    def sample_label_graph(graph_edges, tag2idx, max_neighbors, device):
        sources = []
        targets = []
        for child, parents in graph_edges.items():
            child_id = tag2idx[child]
            k = min(len(parents), max_neighbors)
            parents = random.sample(parents, k)
            for parent in parents:
                parent_id = tag2idx[parent]
                sources.append(child_id)
                targets.append(parent_id)
        tag_graph = dgl.graph((sources, targets))
        tag_graph = dgl.add_self_loop(tag_graph)
        tag_graph = tag_graph.to(device)
        return tag_graph

    def video_encoding(self, x, num_segs):
        B, num_segs = x.shape[:2]
        x = self.video_fc(x)
        # [B, num_segs, in_channels]

        # Multiple Instance Video Learning
        w = torch.tanh(self.mil_P(x)) * torch.sigmoid(self.mil_Q(x))
        # [B, num_segs, in_channels*cluster_size]
        w = w.reshape(B, num_segs, self.cluster_size, -1)
        # [B, num_segs, cluster_size, in_channels]
        w = torch.einsum('btci,ci->btc', w, self.mil_a)
        # [B, num_segs, cluster_size]
        w = F.softmax(w, dim=1)
        # [B, num_segs, cluster_size]
        x = self.mil_dropout(x)
        V = torch.einsum('bti,btc->bci', x, w)
        # [B, cluster_size, in_channels]
        V = self.mil_proj(V)
        # [B, num_classes, in_channels]

        # Label Attention Transformer
        L = F.sigmoid(V)
        # [B, num_classes, in_channels]
        Vf = self.lat_conv(V)
        # [B, 1, in_channels]
        Vf = Vf.squeeze(1)
        # [B, in_channels]
        X = torch.einsum('bci,bi->bci', L, Vf)
        # [B, num_classes, in_channels]
        return X, L

    def tag_encoding(self, X, tag_graph):
        # Label Relation Transformer
        # [B, num_classes, in_channels]
        Z = []
        for x in X:
            # [num_classes, in_channels]
            for gnn, norm in zip(self.tag_gnns, self.gnn_norm):
                x = gnn(tag_graph, x)
                # [num_classes, num_heads, in_channels//num_heads]
                x = x.reshape(self.num_classes, -1)
                x = norm(x)
            # [num_classes, in_channels]
            Z.append(x)
        Z = torch.stack(Z, dim=0)
        # [B, num_classes, in_channels]
        return Z

    def predictor(self, L, Z):
        """Return classification logits"""
        # [B, num_classes, in_channels], [B, num_classes, in_channels]
        cls_score = L + Z
        # [B, num_classes, in_channels]
        if self.dropout is not None:
            cls_score = self.dropout(cls_score)
        cls_score = self.pred(cls_score)
        # [B, num_classes, 1]
        cls_score = cls_score.squeeze(-1)
        # [B, num_classes]
        return cls_score

    def forward(self, x, num_segs):
        X, L = self.video_encoding(x, num_segs)
        # [B, num_classes, in_channels], [B, num_classes, in_channels]
        tag_graph = self.sample_label_graph(
            self.graph_edges, self.tag2idx, 
            self.max_neighbors, X.device)
        Z = self.tag_encoding(X, tag_graph)
        # [B, num_classes, in_channels]
        cls_score = self.predictor(L, Z)
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