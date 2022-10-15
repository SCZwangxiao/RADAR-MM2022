import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax

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
            nn.Linear(video_in_dim, hidden_dim),
            nn.GELU())
        self.tag_embed = nn.Embedding(
            num_tags,
            tag_in_dim)
        self.tag_fc = nn.Sequential(
            nn.BatchNorm1d(tag_in_dim),
            nn.Linear(tag_in_dim, hidden_dim),
            nn.GELU())
        
    def forward(self, mfgs):
        mfgs[0].srcdata['feat']['tag'] += self.tag_embed(mfgs[0].srcdata[dgl.NID]['tag'])
        h = {
            'video': self.video_fc(mfgs[0].srcdata['feat']['video']),
            'tag': self.tag_fc(mfgs[0].srcdata['feat']['tag'])}
        return h


class SimpleHGNLayer(nn.Module):
    def __init__(self,
                 relations,
                 in_dim,
                 out_dim,
                 num_heads=8,
                 dropout=0.2,
                 negative_slope=0.2,
                 use_norm=True):
        super(SimpleHGNLayer, self).__init__()
        self.relations = relations
        self.ntypes = set()
        self.dst_ntypes = {}
        self.etypes = {}
        for srctype, etype, dsttype in relations:
            self.ntypes.add(srctype)
            self.ntypes.add(dsttype)
            if dsttype not in self.dst_ntypes:
                self.dst_ntypes[dsttype] = len(self.dst_ntypes)
            if etype not in self.etypes:
                self.etypes[etype] = len(self.etypes)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = len(relations)
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.use_norm = use_norm

        self.edge_emb = nn.Parameter(torch.FloatTensor(size=(self.num_relations, out_dim)))
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_edge = nn.Linear(out_dim, out_dim, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, self.d_k)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, self.d_k)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, self.d_k)))
        
        if use_norm:
            self.norm = nn.LayerNorm(out_dim)
        self.feat_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        nn.init.xavier_uniform_(self.attn_e)
        nn.init.xavier_uniform_(self.edge_emb)
    
    @staticmethod
    def softmax_among_all(hetgraph, weight_name):
        """Normalize edge weights by softmax across all neighbors.

        Parameters
        -------------
        hetgraph : DGLGraph
            The input heterogeneous graph.
        weight_name : str
            The name of the unnormalized edge weights.
        """
        # Convert to homogeneous graph; DGL will copy the specified data to the new graph.
        g = dgl.to_homogeneous(hetgraph, edata=[weight_name])
        # Call DGL's edge softmax
        g.edata[weight_name] = edge_softmax(g, g.edata[weight_name])
        # Convert it back; DGL again copies the data back to a heterogeneous storage.
        hetg2 = dgl.to_heterogeneous(g, hetgraph.ntypes, hetgraph.etypes)
        # Assign the normalized weights to the original graph
        for etype in hetg2.canonical_etypes:
            hetgraph.edges[etype].data[weight_name] = hetg2.edges[etype].data[weight_name]

    def forward(self, mfg, h):
        h0 = {}
        for ntype in mfg.dsttypes:
            h[ntype] = self.fc(self.feat_drop(h[ntype]))
            h0[ntype] = h[ntype][:mfg.number_of_dst_nodes(ntype)]
        edge_emb = self.fc_edge(self.edge_emb)
        with mfg.local_scope():
            for relation in self.relations:
                srctype, etype, dsttype = relation
                e_id = self.etypes[etype]
                sub_graph = mfg[srctype, etype, dsttype]

                feat_src = h[srctype].view(-1, self.num_heads, self.d_k)
                feat_dst = h0[dsttype].view(-1, self.num_heads, self.d_k)
                feat_edge = edge_emb[e_id].unsqueeze(0).view(1, self.num_heads, self.d_k)
                # compute Simple-HGN attention scores
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                ee = (feat_edge * self.attn_e).sum(dim=-1).unsqueeze(-1)
                er += ee
                sub_graph.srcdata.update({'ft': feat_src, 'el': el})
                sub_graph.dstdata.update({'er': er})
                # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
                sub_graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
                attn = self.leaky_relu(sub_graph.edata.pop('e'))
                sub_graph.edata.update({'attn': attn})
            # compute softmax among all types of neghoures
            self.softmax_among_all(mfg, 'attn')
            # message passing and aggregation
            mfg.multi_update_all(
                {etype : (fn.u_mul_e('ft', 'attn', 'm'), fn.sum('m', 't')) \
                    for etype in self.etypes}, 
                cross_reducer = 'sum')

            new_h = {}
            for ntype, n_id in self.dst_ntypes.items():
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                t = mfg.dstdata['t'][ntype].view(-1, self.out_dim)
                trans_out = self.leaky_relu(t + h0[ntype])
                if self.use_norm:
                    new_h[ntype] = self.norm(trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h 


@GNNS.register_module()
class TagSimpleHGN(nn.Module):
    def __init__(self,
                 num_tags,
                 video_in_dim,
                 tag_in_dim,
                 hidden_dim,
                 layer_relations,
                 video_emb_layer=0,
                 num_heads=1,
                 dropout=0.2,
                 negative_slope=0.2,
                 use_norm=True):
        super(TagSimpleHGN, self).__init__()
        self.num_layers = len(layer_relations)
        self.video_emb_layer = video_emb_layer
        self.common_embedding = CommonEmbedding(
            num_tags,
            video_in_dim,
            tag_in_dim,
            hidden_dim,
            dropout)
        self.layers = nn.ModuleList()
        for relations in layer_relations:
            layer = SimpleHGNLayer(
                relations=relations,
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                negative_slope=negative_slope,
                use_norm=use_norm)
            self.layers.append(layer)
    
    def init_weights(self):
        pass
    
    def forward(self, input_nodes, pair_graph, mfgs):
        h_layer = []
        num_videos = mfgs[-1].number_of_dst_nodes('video')
        h = self.common_embedding(mfgs)

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