import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch.conv import GATConv

from ..builder import GNNS


@GNNS.register_module()
class ClassifierGNN(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        pass
    
    def forward(self, input_nodes, pair_graph, mfgs):
        assert len(mfgs) == 1
        pass
    
    def infer(self, input_nodes, pair_graph, mfgs):
        return self.forward(input_nodes, pair_graph, mfgs)