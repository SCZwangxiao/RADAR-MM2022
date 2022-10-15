import os.path as osp

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class SentenceDataset(BaseDataset):
    """Sentence dataset for text classification.

    The dataset loads raw texts and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        id1    sent1   1
        id2    sent2   1
        id3    sent3   2
        id4    sent4   2
        id5    sent5   3
        id6    sent6   3


    Args:
        ann_file (str): Path to the annotation file.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, **kwargs):
        super().__init__(ann_file, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.rstrip().split('\t') # only strip the '/n' in right, or empty text will be stripped!
                if self.multi_class:
                    assert self.num_classes is not None
                    id, sent, *label = line_split
                    label = list(map(int, label))
                else:
                    id, sent, label = line_split
                    label = int(label)
                video_infos.append(dict(id=id, sents=sent, label=label))
        return video_infos
