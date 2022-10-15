# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import lmdb
import torch
import numpy as np
from mmcv.parallel import DataContainer

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class FrameFeatureDataset(BaseDataset):
    """Frames feature dataset.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, vertical of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        photo_id-1 0 14 343 511
        photo_id-1 1 121 243
        photo_id-1 1 223

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        total_clips (int): Number of extracted frames per video.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        lmdb_tmpl (str): Template for LMDB of each split.
            Default: '{}_lmdb'.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 total_clips,
                 split,
                 data_prefix=None,
                 test_mode=False,
                 io_backend='disk',
                 lmdb_tmpl='{}_lmdb',
                 num_classes=None,
                 feature_organized_by_split=False,
                 start_index=1):
        self.total_clips = total_clips
        assert split in ['train', 'val', 'test']
        self.split = split
        self.io_backend = io_backend
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            True,
            num_classes,
            start_index,
            'RGB',
            sample_by_class=False,
            power=0,
            dynamic_length=False)
        if self.data_prefix is not None:
            self.lmdb_path = osp.join(self.data_prefix, lmdb_tmpl.format(split))
            if feature_organized_by_split:
                self.np_path = osp.join(self.data_prefix, split)
            else:
                self.np_path = self.data_prefix

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                idx = 0
                # idx for photo_id
                photo_id = line_split[idx]
                if photo_id.endswith('mp4'):
                    photo_id = photo_id.split('.')[0]
                video_info['photo_id'] = photo_id
                video_info['total_frames'] = self.total_clips
                idx += 1
                # idx for label[s]
                label = [int(x) for x in line_split[idx:]]
                assert label, f'missing label in line: {line}'
                assert self.num_classes is not None
                video_info['label'] = label
                video_infos.append(video_info)
        return video_infos
    
    def load_feature(self, photo_id):
        if self.io_backend == 'lmdb':
            env = lmdb.open(self.lmdb_path, subdir=osp.isdir(self.lmdb_path),
                            readonly=True, lock=False,
                            readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                byteflow = txn.get(photo_id.encode())
                try:
                    imgs = np.frombuffer(byteflow, dtype=np.float32)
                    imgs = np.copy(imgs)
                except TypeError:
                    print(f'Feature of photo_id {photo_id} not found!')
                    imgs = np.zeros(8*2048)
                imgs = imgs.reshape(self.total_clips, -1)
                imgs = imgs[None,:,:]
        elif self.io_backend == 'disk':
            imgs = np.load(osp.join(self.np_path, f'{photo_id}.npy'))
            imgs = imgs[None,:,:]
        return imgs
    
    def prepare_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        photo_id = results.pop('photo_id')
        imgs = self.load_feature(photo_id)
        results['imgs'] = imgs
        results['modality'] = self.modality
        # prepare tensor in getitem
        results['label'] = DataContainer(
            data=torch.tensor(results['label'])
        )
        return self.pipeline(results)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        return self.prepare_frames(idx)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        return self.prepare_frames(idx)