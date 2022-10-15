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


class HGTLayer(nn.Module):
    def __init__(self,
                 relations,
                 in_dim,
                 out_dim,
                 num_heads=1,
                 dropout = 0.2,
                 use_norm = False):
        super(HGTLayer, self).__init__()
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

        # self.num_ntypes = len(self.ntypes) 
        self.num_relations = len(relations)
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        # self.att = None

        self.k_linears = nn.ModuleDict()
        self.q_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        self.a_linears = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        self.skip = nn.ModuleDict()
        self.use_norm = use_norm
        for ntype in self.ntypes: # Could be simplified
            self.k_linears[ntype] = nn.Linear(in_dim, out_dim)
            self.q_linears[ntype] = nn.Linear(in_dim, out_dim)
            self.v_linears[ntype] = nn.Linear(in_dim, out_dim)
            self.a_linears[ntype] = nn.Linear(out_dim, out_dim)
            if use_norm:
                self.norms[ntype] = nn.LayerNorm(out_dim)
        
        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, num_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, num_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, num_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(len(self.dst_ntypes)))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, mfg, h):
        h0 = {}
        for ntype in mfg.dsttypes:
            h0[ntype] = h[ntype][:mfg.number_of_dst_nodes(ntype)]
        with mfg.local_scope():
            for relation in self.relations:
                srctype, etype, dsttype = relation
                e_id = self.etypes[etype]
                sub_graph = mfg[srctype, etype, dsttype]

                k_linear = self.k_linears[srctype]
                v_linear = self.v_linears[srctype]
                q_linear = self.q_linears[dsttype]

                k = k_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
                q = q_linear(h0[dsttype]).view(-1, self.num_heads, self.d_k)

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%s' % etype] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst') 
                # Here different from that in paper !
                # See fix in https://github.com/dmlc/dgl/issues/2346

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            mfg.multi_update_all(
                {etype : (fn.u_mul_e('v_%s' % etype, 't', 'm'), fn.sum('m', 't')) \
                    for etype in self.etypes}, 
                cross_reducer = 'mean')

            new_h = {}
            for ntype, n_id in self.dst_ntypes.items():
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                alpha = torch.sigmoid(self.skip[n_id])
                t = mfg.dstdata['t'][ntype].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[ntype](t))
                trans_out = trans_out * alpha + h0[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[ntype](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


@GNNS.register_module()
class TagHGT(nn.Module):
    def __init__(self,
                 num_tags,
                 video_in_dim,
                 tag_in_dim,
                 hidden_dim,
                 layer_relations,
                 video_emb_layer=0,
                 num_heads=1,
                 dropout=0.2,
                 use_norm=False):
        super(TagHGT, self).__init__()
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
            layer = HGTLayer(
                relations=relations,
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
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