import torch
from torch import nn
import torch.nn.functional as F

from ..builder import GNNS


@GNNS.register_module()
class DeviseGNN(nn.Module):
    """
    Args:
        node_feat_info (list[tuple]): node feature info.
            Tuple should be in format (str, int, int), with values
            to be node type, in_embed, out embed
    """
    def __init__(self, 
                 video_in_dim,
                 video_out_dim,
                 tag_in_dim,
                 tag_out_dim, 
                 dropout_ratio=.0):
        super().__init__()
        if video_in_dim:
            self.video_fc = nn.Sequential(
                nn.BatchNorm1d(video_out_dim),
                nn.Linear(video_in_dim, video_out_dim),
                nn.Dropout(dropout_ratio),
            )
        else:
            self.video_fc = None
        if tag_in_dim:
            self.tag_fc = nn.Sequential(
                nn.BatchNorm1d(tag_out_dim),
                nn.Linear(tag_in_dim, tag_out_dim),
                nn.Dropout(dropout_ratio),
            )
        else:
            self.tag_fc = None
    
    def init_weights(self):
        pass
    
    def forward(self, input_nodes, pair_graph, mfgs):
        assert len(mfgs) == 1, ('DeviseGnn expect Massage Flow Graph of 1 layer,',
                                'got %d' % len(mfgs))
        video = mfgs[0].dstdata['feat']['video']
        tag = mfgs[0].dstdata['feat']['tag']
        video_h = video + self.video_fc(video) if self.video_fc else video
        tag_h = tag + self.tag_fc(tag) if self.tag_fc else tag
        tag_h = F.normalize(tag_h)
        pair_graph.ndata['h'] = dict(video=video_h, tag=tag_h)
    
    def infer(self, input_nodes, pair_graph, mfgs):
        return self.forward(input_nodes, pair_graph, mfgs)