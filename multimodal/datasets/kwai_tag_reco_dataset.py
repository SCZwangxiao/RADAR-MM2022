import os
import gc
import json
import copy
import os.path as osp
from tqdm import tqdm
from collections import OrderedDict, defaultdict

import mmcv
from mmcv.utils import print_log
import numpy as np
import torch
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

from .builder import DATASETS
from ..core import (mean_average_precision, mmit_mean_average_precision,
                    top_k_recall, top_k_precision)
from .pipelines import Compose


@DATASETS.register_module()
class KwaiTagRecoDataset(DGLDataset):
    """Video feature dataset for video recognition. Reads the features
    extracted off-line. Annotation file can be that of the rawframe dataset,
    or:

    .. code-block:: txt

        id1.mp4\tlabel_idx1\tlabel_idx2
        id2.mp4\tlabel_idx6\tlabel_idx8\tlabel_idx1

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        suffix (str): The suffix of the video feature file. Default: '.npy'.
        kwargs (dict): Other keyword args for `BaseDataset`.
    """

    def __init__(self,
                 vertical,
                 split,
                 dataset_root,
                 video_emb_dir,
                 tag_emb_dir,
                 bidirectional=False,
                 pipeline=None,
                 test_mode=False,
                 force_reload=False,
                 verbose=False):
        assert split in ['train', 'val', 'test']
        self.vertical = vertical
        self.split = split
        self.dataset_root = dataset_root
        self.video_emb_dir = video_emb_dir
        self.tag_emb_dir = tag_emb_dir
        self.bidirectional = bidirectional
        #self.pipeline = Compose(pipeline)
        self.__init_paths__()
        self.test_mode = False if split == 'train' else True
        DGLDataset.__init__(self, name=vertical, force_reload=force_reload, verbose=verbose)
        self.__calculate_basic_attributes__()
    
    def __init_paths__(self):
        self.video_emb_root = osp.join(self.dataset_root, self.video_emb_dir)
        self.tag_emb_root = osp.join(self.dataset_root, self.tag_emb_dir)
        self.ann_file = osp.join(self.dataset_root, f'kwai_{self.vertical}_video_{self.split}_list.txt')
        self.train_ann_file = osp.join(self.dataset_root, f'kwai_{self.vertical}_video_train_list.txt')
        self.ann_class_list = osp.join(self.dataset_root, f'label_map_kwai_{self.vertical}.txt')
        self.tag_parents_root = osp.join(self.dataset_root, f'tag_parents_{self.vertical}.json')
        self.video_children_root = osp.join(self.dataset_root, f'video_children_{self.vertical}.json')
        if not osp.exists(osp.join(self.dataset_root, 'cache')):
            os.mkdir(osp.join(self.dataset_root, 'cache'))
        self.graph_path = osp.join(self.dataset_root, f'cache/{self.vertical}_{self.split}_dgl_graph.bin')
        self.tag_nid2tag_path = osp.join(self.dataset_root, f'cache/{self.vertical}_{self.split}_tag_nid2tag.pkl')
        self.video_nid2pid_path = osp.join(self.dataset_root, f'cache/{self.vertical}_{self.split}_video_nid2pid.pkl')
        self.video_infos_path = osp.join(self.dataset_root, f'cache/{self.vertical}_{self.split}_video_infos.pkl')
    
    def __calculate_basic_attributes__(self):
        self.num_classes = self.g.num_nodes('tag')
        self.num_videos = self.g.num_nodes('video')
        if self.test_mode:
            self.num_test_videos = self.g.num_edges(etype=('tag', 'WhetherHasVideo', 'video')) // self.num_classes
            self.num_train_videos = self.num_videos - self.num_test_videos
        if self.bidirectional:
            SubTopic_src, SubTopic_dst = self.g.edges(etype=('tag', 'SubTopic', 'tag'))
            NotHasVideo_src, NotHasVideo_dst = self.g.edges(etype=('tag', 'NotHasVideo', 'video'))
            HasVideo_src, HasVideo_dst = self.g.edges(etype=('tag', 'HasVideo', 'video'))
            HasTag_src, HasTag_dst = self.g.edges(etype=('video', 'HasTag', 'tag'))
            FollowedBy_src, FollowedBy_dst = self.g.edges(etype=('video', 'FollowedBy', 'video'))
            video_feats = self.g.nodes['video'].data['feat']
            tag_feats = self.g.nodes['tag'].data['feat']
            graph_data = {
                ('tag', 'SubTopic', 'tag'): (SubTopic_src, SubTopic_dst),
                ('tag', 'SuperTopic', 'tag'): (SubTopic_dst, SubTopic_src),
                ('tag', 'NotHasVideo', 'video'): (NotHasVideo_src, NotHasVideo_dst),
                ('tag', 'HasVideo', 'video'): (HasVideo_src, HasVideo_dst),
                ('video', 'HasTag', 'tag'): (HasTag_src, HasTag_dst),
                ('video', 'FollowedBy', 'video'): (FollowedBy_src, FollowedBy_dst)
            }
            if self.test_mode:
                WhetherHasVideo_src, WhetherHasVideo_dst = self.g.edges(etype=('tag', 'WhetherHasVideo', 'video'))
                graph_data[('tag', 'WhetherHasVideo', 'video')] = (WhetherHasVideo_src, WhetherHasVideo_dst)
            g = dgl.heterograph(graph_data, idtype=torch.int32)
            g.nodes['video'].data['feat'] = video_feats
            g.nodes['tag'].data['feat'] = tag_feats
            self.g = g
    
    @staticmethod
    def get_nid(nid2ele, ele2nid, tag):
        if tag in ele2nid:
            return ele2nid[tag]
        else:
            nid = len(ele2nid)
            ele2nid[tag] = nid
            nid2ele[nid] = tag
            return nid
    
    def process(self):
        print(f'Start building {self.vertical} {self.split} set ...')
        # process raw data to graphs
        # Node info
        tag_nid2tag = {}
        tag2nid = {}
        video_nid2pid = {}
        pid2nid = {}
        video_infos = []
        tag_feats = []
        video_feats = []
        # edge info
        SubTopic_src = []
        SubTopic_dst = []
        NotHasVideo_src = []
        NotHasVideo_dst = []
        HasVideo_src = []
        HasVideo_dst = []
        HasTag_src = []
        HasTag_dst = []
        FollowedBy_src = []
        FollowedBy_dst = []
        WhetherHasVideo_src = []
        WhetherHasVideo_dst = []
        # Load video labels
        with open(self.ann_class_list, 'r') as F:
            lines = F.readlines()
        for idx, line in tqdm(enumerate(lines), desc='Loading video labels ...'):
            tag = line.strip()
            tag_nid2tag[idx] = tag
            tag2nid[tag] = idx
        # Load tag hierarchy
        with open(self.tag_parents_root, 'r') as F:
            tag_parents = json.load(F)
        for tag, parents in tqdm(tag_parents.items(), desc='Loading tag hierarchy ...'):
            self.get_nid(tag_nid2tag, tag2nid, tag)
            tag_nid = tag2nid[tag]
            for parent in parents:
                self.get_nid(tag_nid2tag, tag2nid, parent)
                parent_nid = tag2nid[parent]
                SubTopic_src.append(tag_nid)
                SubTopic_dst.append(parent_nid)
        # Load video - tag annotations
        with open(self.train_ann_file, 'r') as F:
            lines = F.readlines()
        for line in tqdm(lines, desc='Loading video - tag annotations ...'):
            pid, *tag_ids = line.strip().split('\t')
            pid = int(pid.split('.')[0])
            tag_ids = list(map(int, tag_ids))
            if not self.test_mode:
                # In test mode we should store labels of test videos
                video_infos.append(dict(label=tag_ids))
            self.get_nid(video_nid2pid, pid2nid, pid)
            video_nid = pid2nid[pid]
            for tag_nid in tag_ids:
                HasVideo_src.append(tag_nid)
                HasVideo_dst.append(video_nid)
                HasTag_src.append(video_nid)
                HasTag_dst.append(tag_nid)
            tag_ids = set(tag_ids)
            for tag_nid in tag_nid2tag:
                if tag_nid not in tag_ids:
                    NotHasVideo_src.append(tag_nid)
                    NotHasVideo_dst.append(video_nid)
        if self.test_mode:
            # load testing videos
            with open(self.ann_file, 'r') as F:
                lines = F.readlines()
            all_tag_ids = list(range(len(tag_nid2tag)))
            for line in lines:
                pid, *tag_ids = line.strip().split('\t')
                pid = int(pid.split('.')[0])
                tag_ids = list(map(int, tag_ids))
                video_infos.append(dict(label=tag_ids))
                assert pid not in pid2nid
                self.get_nid(video_nid2pid, pid2nid, pid)
                video_nid = pid2nid[pid]
                # In test mode, we should predict the probability against all tags
                WhetherHasVideo_src.extend(all_tag_ids)
                WhetherHasVideo_dst.extend([video_nid]*len(all_tag_ids))
        # Load video - video relations
        with open(self.video_children_root, 'r') as F:
            video_children = json.load(F)
        for pid in tqdm(pid2nid, desc='Loading video - video relations ...'):
            children = video_children.get(str(pid), [])
            parent_nid = pid2nid[pid]
            for child_pid in children:
                if child_pid not in pid2nid:
                    continue
                child_nid = pid2nid[child_pid]
                FollowedBy_src.append(parent_nid)
                FollowedBy_dst.append(child_nid)
        del tag_parents, video_children, lines
        gc.collect()
        # Load video features
        for video_nid in tqdm(range(len(pid2nid)), desc='Loading video features ...'):
            pid = video_nid2pid[video_nid]
            feat_filename = osp.join(self.video_emb_root, str(pid)) + '.npy'
            feat = np.load(feat_filename)
            feat = torch.tensor(feat)
            video_feats.append(feat)
        video_feats = torch.stack(video_feats)
        gc.collect()
        # Load tag features
        for tag_nid in tqdm(range(len(tag2nid)), desc='Loading tag features ...'):
            tag = tag_nid2tag[tag_nid]
            feat_filename = osp.join(self.tag_emb_root, tag) + '.npy'
            feat = np.load(feat_filename)
            feat = torch.tensor(feat)
            tag_feats.append(feat)
        tag_feats = torch.stack(tag_feats)
        gc.collect()
        # Build graph
        gc.collect()
        graph_data = {
            ('tag', 'SubTopic', 'tag'): (torch.tensor(SubTopic_src), torch.tensor(SubTopic_dst)),
            ('tag', 'NotHasVideo', 'video'): (torch.tensor(NotHasVideo_src), torch.tensor(NotHasVideo_dst)),
            ('tag', 'HasVideo', 'video'): (torch.tensor(HasVideo_src), torch.tensor(HasVideo_dst)),
            ('video', 'HasTag', 'tag'): (torch.tensor(HasTag_src), torch.tensor(HasTag_dst)),
            ('video', 'FollowedBy', 'video'): (torch.tensor(FollowedBy_src), torch.tensor(FollowedBy_dst))
        }
        if self.test_mode:
            graph_data[('tag', 'WhetherHasVideo', 'video')] = (torch.tensor(WhetherHasVideo_src), torch.tensor(WhetherHasVideo_dst))
        print('Building graph ...')
        g = dgl.heterograph(graph_data, idtype=torch.int32)
        gc.collect()
        print('Assign features ...')
        g.nodes['video'].data['feat'] = video_feats
        g.nodes['tag'].data['feat'] = tag_feats
        gc.collect()
        # Assign
        self.g = g
        self.tag_nid2tag = tag_nid2tag
        self.video_nid2pid = video_nid2pid
        self.video_infos = video_infos

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_graph(idx)
        return self.prepare_train_graph(idx)

    def __len__(self):
        if self.test_mode:
            return self.num_test_videos
        return self.num_videos
    
    def save(self):
        # save processed data
        print('Saveing processed dataset ...')
        save_graphs(self.graph_path, [self.g])
        save_info(self.tag_nid2tag_path, self.tag_nid2tag)
        save_info(self.video_nid2pid_path, self.video_nid2pid)
        save_info(self.video_infos_path, self.video_infos)
    
    def load(self):
        # load processed data
        print('Loading processed dataset ...')
        self.g = load_graphs(self.graph_path)[0][0]
        self.tag_nid2tag = load_info(self.tag_nid2tag_path)
        self.video_nid2pid = load_info(self.video_nid2pid_path)
        self.video_infos = load_info(self.video_infos_path)
    
    def has_cache(self):
        # check whether there are processed data
        return osp.exists(self.graph_path) and osp.exists(self.tag_nid2tag_path) and osp.exists(self.video_nid2pid_path) and osp.exists(self.video_infos_path)
    
    def prepare_train_graph(self, idx):
        return idx
    
    def prepare_test_graph(self, idx):
        # testing videos are ranged behind training videos
        return self.num_train_videos + idx
    
    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr
    
    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'mean_average_precision', 'mmit_mean_average_precision', 
            'top_k_recall', 'top_k_precision'
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric in [
                    'mean_average_precision', 'mmit_mean_average_precision',
            ]:
                gt_labels_multilabel = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(results, gt_labels_multilabel)
                    eval_results['mean_average_precision'] = mAP
                    log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(results, gt_labels_multilabel)
                    eval_results['mmit_mean_average_precision'] = mAP
                    log_msg = f'\nmmit_mean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue
                
            if metric == 'top_k_recall':
                gt_labels_multilabel = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                topk = metric_options.setdefault('top_k_recall',
                                                 {}).setdefault(
                                                     'topk', (5, ))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )
                
                top_k_rec = top_k_recall(results, gt_labels_multilabel, topk)
                log_msg = []
                for k, recall in zip(topk, top_k_rec):
                    eval_results[f'top{k}_recall'] = recall
                    log_msg.append(f'\ntop{k}_recall\t{recall:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue
                
            if metric == 'top_k_precision':
                gt_labels_multilabel = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                topk = metric_options.setdefault('top_k_precision',
                                                 {}).setdefault(
                                                     'topk', (1, ))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )
                
                top_k_prec = top_k_precision(results, gt_labels_multilabel, topk)
                log_msg = []
                for k, precision in zip(topk, top_k_prec):
                    eval_results[f'top{k}_precision'] = precision
                    log_msg.append(f'\ntop{k}_precision\t{precision:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

        return eval_results

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)