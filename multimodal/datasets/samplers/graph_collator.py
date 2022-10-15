import torch
import dgl


class KwaiNodeCollator():
    def __init__(self, g, block_sampler, test_mode):
        self.g = g
        self.block_sampler = block_sampler
        self.test_mode = test_mode
        self.pos_etype = ('tag', 'HasVideo', 'video')
        self.neg_etype = ('tag', 'NotHasVideo', 'video')
        self.infer_etype = ('tag', 'WhetherHasVideo', 'video')
    
    def collate(self, video_nids):
        if self.test_mode:
            return self.collate_infer(video_nids)
        return self.collate_train(video_nids)
    
    def collate_train(self, video_nids):
        items = {'video': torch.tensor(video_nids, dtype=self.g.idtype)}
        # Sample pos & neg graph
        pos_pair_eids = self.g.in_edges(items['video'], form='eid', etype=self.pos_etype)
        neg_pair_eids = self.g.in_edges(items['video'], form='eid', etype=self.neg_etype)
        pair_graph = self.g.edge_subgraph({self.pos_etype: pos_pair_eids, 
                                           self.neg_etype: neg_pair_eids})
        # no need to apply transform.compact_graphs() since there can't be isolated nodes
        # Sample MFGs
        seed_nodes = pair_graph.ndata[dgl.NID]
        blocks = self.block_sampler.sample_blocks(self.g, seed_nodes)
        input_nodes = blocks[0].srcdata[dgl.NID]
        pair_graph.ndata.pop('feat')
        batch = {
            'input_nodes': input_nodes, 
            'pair_graph': pair_graph, 
            'mfgs': blocks
        }
        return batch
    
    def collate_infer(self, video_nids):
        items = {'video': torch.tensor(video_nids, dtype=self.g.idtype)}
        # Sample pos & neg graph
        unknown_pair_eids = self.g.in_edges(items['video'], form='eid', etype=self.infer_etype)
        unknown_pair_graph = self.g.edge_subgraph({self.infer_etype: unknown_pair_eids})
        unknown_pair_graph.ndata.pop('feat')
        # no need to apply transform.compact_graphs() since there can't be isolated nodes
        # Sample MFGs
        seed_nodes = unknown_pair_graph.ndata[dgl.NID]
        blocks = self.block_sampler.sample_blocks(self.g, seed_nodes)
        input_nodes = blocks[0].srcdata[dgl.NID]
        batch = {
            'input_nodes': input_nodes, 
            'pair_graph': unknown_pair_graph, 
            'mfgs': blocks
        }
        return batch