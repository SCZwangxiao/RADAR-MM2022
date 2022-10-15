from copy import deepcopy
import os.path as osp

import torch
from mmcv.utils import build_from_cfg

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class MultimodalDataset(BaseDataset):
    """Multimodal dataset for video recognition. Dataset is constructed by using 
    configs of each modality. Reads the features by using the pipline of each modality. 

    Args:
        types_modality (list<str>): Path to the annotation file.
        ann_file (list<str>): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        modality (list<str>): Modality of data. Support 'RGB', 'Text', 'Audio'.
        use_feat (list<bool>): Whether to load pre-extrected feature. Build 
            "video_feat_path", "text_feat_path", and "audio_feat_path" if enabled.
        feat_suffix (list<str>): The suffix of the feature file. Default: '.pkl'.
        kwargs (dict): Other keyword args for `BaseDataset`.
    """

    def __init__(self, types_modality, ann_file, pipeline, modality, use_feat=True, feat_suffix='.pkl', **kwargs):
        self.num_modalities = len(types_modality)
        assert self.num_modalities == len(ann_file)
        assert self.num_modalities == len(modality)
        if type(use_feat) is not list:
            use_feat = [use_feat] * self.num_modalities
        if type(feat_suffix) is not list:
            feat_suffix = [feat_suffix] * self.num_modalities

        temp = deepcopy(kwargs)
        temp.pop('data_prefix') # data_prefix must be str or None, cannot be list
        super().__init__(ann_file, pipeline, modality=modality, **temp)

        self.use_feat = use_feat
        self.feat_suffix = feat_suffix
        self.data_prefix = kwargs['data_prefix']
        # Merge annotation of each modality
        for i in range(self.num_modalities):
            cfg = dict(
                type=types_modality[i],
                ann_file=ann_file[i],
                pipeline=pipeline,
                modality=modality[i],
            )
            # Pass other params.
            for key, value in kwargs.items():
                if type(value) is list:
                    # Here we must assert none of the params are list
                    assert len(value) == self.num_modalities
                    if value[i] is None:  # None stands for not exits
                        continue
                    cfg[key] = value[i]
                else:
                    cfg[key] = value
            dataset = build_from_cfg(cfg, DATASETS)
            self.__merge_annotations(dataset.video_infos)
        self.__modify_path_for_feature()

    def __merge_annotations(self, new_video_infos):
        if len(self.video_infos) == 0:
            self.video_infos = new_video_infos
        else:
            for idx in range(len(self.video_infos)):
                video_info = self.video_infos[idx]
                for k, v in new_video_infos[idx].items():
                    if k in video_info:
                        assert v == video_info[k], (
                            f'video information of {idx}th sample differs in {k}, '
                            f'expected {v}, got {video_info[k]}.'
                        )
                    else:
                        video_info[k] = v
                self.video_infos[idx] = video_info

    def __modify_path_for_feature(self):
        for mid in range(self.num_modalities):
        #for modality, use_feat, feat_suffix in zip(self.modality, self.use_feat, self.feat_suffix):
            if not self.use_feat[mid]:
                continue
            data_prefix = self.data_prefix[mid]
            feat_suffix = self.feat_suffix[mid]
            for idx in range(len(self.video_infos)):
                if self.modality[mid] == 'RGB':
                    filename = self.video_infos[idx]['filename']
                    id = osp.basename(filename).split('.')[0]
                    video_feat_path = osp.join(data_prefix, id) + feat_suffix
                    self.video_infos[idx]['video_feat_path'] = video_feat_path
                elif self.modality[mid] == 'Text':
                    id = self.video_infos[idx]['id']
                    text_feat_path = osp.join(data_prefix, id) + feat_suffix
                    self.video_infos[idx]['text_feat_path'] = text_feat_path
                elif self.modality[mid] == 'Audio':
                    raise NotImplementedError
                else:
                    raise NotImplementedError

    def load_annotations(self):
        return []
