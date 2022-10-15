import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from mmcv.utils import Registry
from mmcv.cnn import MODELS as MMCV_MODELS

from ..builder import GNNS


GNN_MODULES = Registry('gnns')
MASSAGE_PASSER = GNN_MODULES
AGGREGATOR = GNN_MODULES
UPDATOR = GNN_MODULES
REDUCER = GNN_MODULES

class CommonEmbedding(nn.Module):
    def __init__(self,
                 num_tags,
                 video_in_dim,
                 tag_in_dim,
                 hidden_dim):
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


@MASSAGE_PASSER.register_module()
class TransformerMessagePasser(nn.Module):
    def __init__(self,
                 relations,
                 in_dim,
                 out_dim,
                 num_heads):
        super().__init__()
        self.relations = relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.get_node_edge_types(relations)

        self.k_linears = nn.ModuleDict()
        self.q_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        self.a_linears = nn.ModuleDict()
        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, num_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, num_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, num_heads, self.d_k, self.d_k))

        for srctype in self.src_ntypes:
            self.k_linears[srctype] = nn.Linear(in_dim, out_dim)
            self.v_linears[srctype] = nn.Linear(in_dim, out_dim)
        for dsttype in self.dst_ntypes:
            self.q_linears[dsttype] = nn.Linear(in_dim, out_dim)

        # Init
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def get_node_edge_types(self, relations):
        self.src_ntypes = set()
        self.dst_ntypes = {}
        self.etypes = {}
        for srctype, etype, dsttype in relations:
            self.src_ntypes.add(srctype)
            self.dst_ntypes[dsttype] = self.dst_ntypes.get(dsttype, len(self.dst_ntypes))
            self.etypes[etype] = self.etypes.get(etype, len(self.etypes))
        self.num_relations = len(relations)
        self.num_dst_types = len(self.dst_ntypes)

    def followedby_message(self, sub_graph, h, h0, return_logits=False):
        srctype, etype, dsttype = ('video', 'FollowedBy', 'video')
        e_id = self.etypes[etype]
        k_linear = self.k_linears[srctype]
        v_linear = self.v_linears[srctype]
        q_linear = self.q_linears[dsttype]
        relation_att = self.relation_att[e_id]
        relation_pri = self.relation_pri[e_id]
        relation_msg = self.relation_msg[e_id]

        k = k_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
        v = v_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
        q = q_linear(h0[dsttype]).view(-1, self.num_heads, self.d_k)
        k = torch.einsum("bij,ijk->bik", k, relation_att)
        v = torch.einsum("bij,ijk->bik", v, relation_msg)
        sub_graph.srcdata['k'] = k
        sub_graph.dstdata['q'] = q
        sub_graph.srcdata['v_%s' % etype] = v

        sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
        attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
        sub_graph.edata['t'] = attn_score
        # attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')
        # sub_graph.edata['t'] = attn_score.unsqueeze(-1)

    def subtopic_message(self, sub_graph, h, h0):
        srctype, etype, dsttype = ('tag', 'SubTopic', 'tag')
        e_id = self.etypes[etype]
        k_linear = self.k_linears[srctype]
        v_linear = self.v_linears[srctype]
        q_linear = self.q_linears[dsttype]
        relation_att = self.relation_att[e_id]
        relation_pri = self.relation_pri[e_id]
        relation_msg = self.relation_msg[e_id]

        k = k_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
        v = v_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
        q = q_linear(h0[dsttype]).view(-1, self.num_heads, self.d_k)
        k = torch.einsum("bij,ijk->bik", k, relation_att)
        v = torch.einsum("bij,ijk->bik", v, relation_msg)
        sub_graph.srcdata['k'] = k
        sub_graph.dstdata['q'] = q
        sub_graph.srcdata['v_%s' % etype] = v

        sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
        attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
        sub_graph.edata['t'] = attn_score
        # attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')
        # sub_graph.edata['t'] = attn_score.unsqueeze(-1)

    def hastag_message(self, sub_graph, h, h0):
        srctype, etype, dsttype = ('video', 'HasTag', 'tag')
        e_id = self.etypes[etype]
        k_linear = self.k_linears[srctype]
        v_linear = self.v_linears[srctype]
        q_linear = self.q_linears[dsttype]
        relation_att = self.relation_att[e_id]
        relation_pri = self.relation_pri[e_id]
        relation_msg = self.relation_msg[e_id]

        k = k_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
        v = v_linear(h[srctype]).view(-1, self.num_heads, self.d_k)
        q = q_linear(h0[dsttype]).view(-1, self.num_heads, self.d_k)
        k = torch.einsum("bij,ijk->bik", k, relation_att)
        v = torch.einsum("bij,ijk->bik", v, relation_msg)
        sub_graph.srcdata['k'] = k
        sub_graph.dstdata['q'] = q
        sub_graph.srcdata['v_%s' % etype] = v

        sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
        attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
        sub_graph.edata['t'] = attn_score
        # attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')
        # sub_graph.edata['t'] = attn_score.unsqueeze(-1)
    
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
        g.edata[weight_name] = edge_softmax(g, g.edata[weight_name]) ## .unsqueeze(-1)
        # Convert it back; DGL again copies the data back to a heterogeneous storage.
        hetg2 = dgl.to_heterogeneous(g, hetgraph.ntypes, hetgraph.etypes)
        # Assign the normalized weights to the original graph
        for etype in hetg2.canonical_etypes:
            hetgraph.edges[etype].data[weight_name] = hetg2.edges[etype].data[weight_name]

    def forward(self, mfg, h, h0):
        for relation in self.relations:
            srctype, etype, dsttype = relation
            sub_graph = mfg[srctype, etype, dsttype]
            if etype == 'FollowedBy':
                self.followedby_message(sub_graph, h, h0)
            elif etype == 'SubTopic':
                self.subtopic_message(sub_graph, h, h0)
            elif etype == 'HasTag':
                self.hastag_message(sub_graph, h, h0)
            else:
                raise NotImplementedError
        self.softmax_among_all(mfg, 't')


@AGGREGATOR.register_module()
class StackAggregator(nn.Module):
    def __init__(self,
                 relations):
        super().__init__()
        self.etypes = {}
        for srctype, etype, dsttype in relations:
            self.etypes[etype] = self.etypes.get(etype, len(self.etypes))

    def forward(self, mfg, h, h0):
        mfg.multi_update_all(
            {etype : (fn.u_mul_e('v_%s' % etype, 't', 'm'), fn.sum('m', 'hm')) \
                for etype in self.etypes}, 
            cross_reducer = 'stack')


@UPDATOR.register_module()
class SumUpdator(nn.Module):
    def __init__(self,
                 relations,
                 hidden_dim,
                 dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.get_node_edge_types(relations)

        self.dropout = nn.Dropout(dropout)

    def get_node_edge_types(self, relations):
        self.dst_ntypes = {}
        for srctype, etype, dsttype in relations:
            self.dst_ntypes[dsttype] = self.dst_ntypes.get(dsttype, len(self.dst_ntypes))

    def forward(self, mfg, h, h0):
        trans_out = {}
        for ntype, n_id in self.dst_ntypes.items():
            hm = mfg.dstdata['hm'][ntype]
            n_intype = hm.shape[1]
            hm = hm.view(-1, n_intype, self.hidden_dim)
            h0_dst = h0[ntype].unsqueeze(1).expand(-1, n_intype, -1)
            trans_out[ntype] = hm + h0_dst
        return trans_out


@REDUCER.register_module()
class CommonUniqueReducer(nn.Module):
    def __init__(self,
                 relations,
                 hidden_dim,
                 dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.get_node_edge_types(relations)

        self.norms = nn.ModuleDict()
        for dsttype in self.dst_ntypes:
            self.norms[dsttype] = nn.LayerNorm(hidden_dim)
        
        if 'HasTag' in self.etypes and 'SubTopic' in self.etypes:
            self.dropout = nn.Dropout(dropout)
            self.u1_fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU())
            self.u2_fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU())
            self.c_fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU())
            self.fusion = nn.Linear(hidden_dim, hidden_dim)

    def get_node_edge_types(self, relations):
        self.dst_ntypes = {}
        self.etypes = {}
        for srctype, etype, dsttype in relations:
            self.dst_ntypes[dsttype] = self.dst_ntypes.get(dsttype, len(self.dst_ntypes))
            self.etypes[etype] = self.etypes.get(etype, len(self.etypes))

    def common_unique(self, trans_out):
        # [N, 2, d]
        trans_out1 = self.dropout(trans_out[:,0,:])
        trans_out2 = self.dropout(trans_out[:,1,:])
        # [N, d]
        unique1 = self.u1_fc(trans_out1)
        unique2 = self.u2_fc(trans_out2)
        common1 = self.c_fc(trans_out1)
        common2 = self.c_fc(trans_out2)
        # [N, d]
        unique = torch.stack([unique1, unique2], dim=1)
        common = torch.stack([common1, common2], dim=1)
        # [N, 2, d]
        # trans_out = (unique1 + common.mean(1) + unique2) / 3
        trans_out = (common.mean(1) + unique.max(1).values) / 2
        trans_out = self.fusion(trans_out)
        # [N, d]
        return trans_out, unique, common

    def forward(self, mfg, trans_outs, h0):
        h_new = {}
        h_unique = {}
        h_common = {}
        for ntype, trans_out in trans_outs.items():
            if trans_out.shape[1] == 1:
                # Only one inbound edge etype
                trans_out = trans_out.mean(1)
            else:
                trans_out, unique, common = self.common_unique(trans_out)
                h_unique[ntype] = unique
                h_common[ntype] = common
            h_new[ntype] = self.norms[ntype](trans_out)
        result_dict = {
            'h_new': h_new,
            'h_unique': h_unique,
            'h_common': h_common}
        return result_dict


def build_message_passer(message_passer, relations):
    message_passer['relations'] = relations
    return MASSAGE_PASSER.build(message_passer)


def build_aggregator(aggregator, relations):
    aggregator['relations'] = relations
    return AGGREGATOR.build(aggregator)


def build_updator(updator, relations):
    updator['relations'] = relations
    return UPDATOR.build(updator)


def build_reducer(reducer, relations):
    reducer['relations'] = relations
    return REDUCER.build(reducer)


class GNNLayer(nn.Module):
    def __init__(self,
                 relations,
                 message_passer,
                 aggregator,
                 updator,
                 reducer):
        super().__init__()
        self.message_passer = build_message_passer(message_passer, relations)
        self.aggregator = build_aggregator(aggregator, relations)
        self.updator = build_updator(updator, relations)
        self.reducer = build_reducer(reducer, relations)
    
    def makeup_non_updated_nodes(self, result_dict, mfg, h0):
        h_new = result_dict['h_new']
        for ntype, h0_src in h0.items():
            if ntype not in h_new:
                h_new[ntype] = h0_src
        result_dict['h_new'] = h_new
        return result_dict

    def forward(self, mfg, h):
        h0 = {}
        for ntype in mfg.dsttypes:
            # h0 in dst nodes
            h0[ntype] = h[ntype][:mfg.num_dst_nodes(ntype)]
        with mfg.local_scope():
            # Message Passing
            self.message_passer(mfg, h, h0)
            # Aggregation
            self.aggregator(mfg, h, h0)
            # Update
            trans_out = self.updator(mfg, h, h0)
            # Reduce
            result_dict = self.reducer(mfg, trans_out, h0)
            result_dict = self.makeup_non_updated_nodes(result_dict, mfg, h0)
        h_new = result_dict.pop('h_new')
        for key, v in result_dict.items():
            mfg.ndata[key] = v
        return h_new


@GNNS.register_module()
class OurMutualGNN(nn.Module):
    def __init__(self,
                 num_tags,
                 video_in_dim,
                 tag_in_dim,
                 hidden_dim,
                 video_emb_layer,
                 layer_relations,
                 message_passer,
                 aggregator,
                 updator,
                 reducer):
        super().__init__()
        self.num_layers = len(layer_relations)
        self.video_emb_layer = video_emb_layer
        self.common_embedding = CommonEmbedding(
            num_tags,
            video_in_dim,
            tag_in_dim,
            hidden_dim)
        self.layers = nn.ModuleList()
        for relations in layer_relations:
            layer = GNNLayer(
                relations=relations,
                message_passer=message_passer,
                aggregator=aggregator,
                updator=updator,
                reducer=reducer)
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

    def infer_get_network_details(self, input_nodes, pair_graph, mfgs):
        """
        Get gate score and attention logits in layer 1
        """
        # Get gate score
        num_videos = mfgs[-1].number_of_dst_nodes('video')
        h = self.common_embedding(mfgs)
        mfg = mfgs[0]
        gnn = self.layers[0]
        h0 = {}
        for ntype in mfg.dsttypes:
            h0[ntype] = h[ntype][:mfg.number_of_dst_nodes(ntype)]
        attn_score_logits = gnn.message_passer.followedby_message(mfg['video', 'FollowedBy', 'video'], h, h0, return_logits=True)
        gnn.message_passer(mfg, h, h0)
        gnn.aggregator(mfg, h, h0)
        delta_z = gnn.updator.homo_updator.get_delta_z(mfg, h, h0)
        z = delta_z['video']
        return attn_score_logits, z
