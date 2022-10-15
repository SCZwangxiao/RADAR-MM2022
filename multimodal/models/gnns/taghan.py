import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch.conv import GATConv

from ..builder import GNNS


class CommonEmbedding(nn.Module):
    def __init__(self,
                 num_tags,
                 video_in_dim,
                 tag_in_dim,
                 hidden_dim,
                 feat_drop):
        super().__init__()
        self.video_fc = nn.Sequential(
            nn.BatchNorm1d(video_in_dim),
            nn.Linear(video_in_dim, hidden_dim))
        self.tag_embed = nn.Embedding(
            num_tags,
            tag_in_dim)
        self.tag_fc = nn.Sequential(
            nn.BatchNorm1d(tag_in_dim),
            nn.Linear(tag_in_dim, hidden_dim))
        
    def forward(self, mfgs):
        mfgs[0].srcdata['feat']['tag'] += self.tag_embed(mfgs[0].srcdata[dgl.NID]['tag'])
        mfgs[0].srcdata['feat'] = {
            'video': self.video_fc(mfgs[0].srcdata['feat']['video']),
            'tag': self.tag_fc(mfgs[0].srcdata['feat']['tag'])}
        return mfgs


class GAT(GATConv):
    def __init__(self,
                 in_feats, 
                 out_feats, 
                 num_heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 activation):
        super().__init__(
            in_feats=in_feats, 
            out_feats=out_feats//num_heads, 
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope)
        self.h_dim = out_feats
        self.fc = nn.Linear(out_feats, out_feats)
        self.activation = getattr(F, activation)

    def aggregate_and_update(self, h0, hm):
        hm = hm.reshape(-1, self.h_dim)
        h1 = self.fc(h0) + self.activation(hm)
        return h1

    def forward(self, graph, feat, get_attention=False):
        if not graph.is_block:
            raise NotImplementedError

        h0_dst = feat[1][:graph.number_of_dst_nodes()]
        if get_attention:
            hm, att = super().forward(graph, feat, get_attention)
        else:
            hm = super().forward(graph, feat, get_attention)
        h1 = self.aggregate_and_update(h0_dst, hm)

        if get_attention:
            return h1, att
        else:
            return h1


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, inputs, dsttype):
        z = torch.stack(inputs, dim=1)
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)


@GNNS.register_module()
class TagHAN(nn.Module):
    def __init__(self,
                 num_tags,
                 video_in_dim,
                 tag_in_dim,
                 hidden_dim,
                 layer_etypes,
                 video_emb_layer=0,
                 num_heads=1,
                 feat_drop=0.5,
                 attn_drop=0.,
                 negative_slope=0.2):
        super().__init__()
        self.num_layers = len(layer_etypes)
        self.video_emb_layer = video_emb_layer
        self.hidden_dim = hidden_dim

        self.common_embedding = CommonEmbedding(
            num_tags,
            video_in_dim,
            tag_in_dim,
            hidden_dim,
            feat_drop)
        self.layers = nn.ModuleList()
        for etypes in layer_etypes:
            mods = {}
            for etype in etypes:
                mods[etype] = GAT(
                    in_feats=hidden_dim, 
                    out_feats=hidden_dim, 
                    num_heads=num_heads,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    activation='relu')
            aggregate_fn = SemanticAttention(
                in_size=hidden_dim)
            conv_layer = HeteroGraphConv(
                mods=mods,
                aggregate=aggregate_fn)
            self.layers.append(conv_layer)

    def init_weights(self):
        pass
    
    def forward(self, input_nodes, pair_graph, mfgs):
        h_layer = []
        num_videos = mfgs[-1].number_of_dst_nodes('video')
        mfgs = self.common_embedding(mfgs)

        h = mfgs[0].srcdata['feat']
        h_layer.append(h) # h0
        for l in range(self.num_layers):
            mfg = mfgs[l]
            h = self.layers[l](mfg, h)
            h_layer.append(h)
        h['video'] = h_layer[self.video_emb_layer]['video'][:num_videos]
        h['video'] = F.normalize(h['video'])
        pair_graph.ndata['h'] = h
    
    def infer(self, input_nodes, pair_graph, mfgs):
        return self.forward(input_nodes, pair_graph, mfgs)
    