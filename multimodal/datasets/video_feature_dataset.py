import os.path as osp

import torch

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class VideoFeatureDataset(BaseDataset):
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

    def __init__(self, ann_file, pipeline, suffix='.npy', **kwargs):
        self.suffix = suffix
        super().__init__(ann_file, pipeline, modality='RGB', **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                video_info = {}
                line_split = line.strip().split()
                filename, label = line_split[0], line_split[1:]
                # Get feature filename
                if self.data_prefix is not None:
                    if filename.endswith('.mp4'):
                        filename = filename[:-4]
                        filename = osp.join(self.data_prefix,
                                            filename) + self.suffix
                    elif not filename.endswith(self.suffix):
                        filename = osp.join(self.data_prefix,
                                            filename) + self.suffix
                    else:
                        filename = osp.join(self.data_prefix, filename)
                video_info['video_feat_path'] = filename
                # Get label
                if self.multi_class:
                    assert self.num_classes is not None
                    label = list(map(int, label))
                    video_info['label'] = label
                else:
                    video_info['label'] = int(label)
                video_infos.append(video_info)
        return video_infos
