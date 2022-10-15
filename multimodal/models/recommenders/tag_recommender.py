import warnings
from collections import OrderedDict

import torch
from torch import nn
import torch.distributed as dist

from .. import builder
from ..builder import RECOMMENDERS


class TagTower(nn.Module):
    """Tag tower
    Args:
        num_tags (int): num of tags.
        mebed_dim (int): dimension of tag embedding
        data_prefix (str): path to pretrain files， None for random init。
        tag_map_file (str): path to file in the following format:
            [line1]tag1
            [line1]tag2
            [lineN]...
        tag_relation_file (str): path to tag relation json file:
            {tag1: [parent1, parent2, ...]}
    """
    def __init__(self,
                 num_tags,
                 embed_dim,
                 data_prefix=None,
                 freeze=False,
                 tag_map_file=None,
                 tag_relation_file=None):
        super().__init__()
        self.num_tags = num_tags
        self.data_prefix = data_prefix
        self.freeze = freeze
        self.tag_map_file = tag_map_file
        self.tag_relation_file = tag_relation_file

        self.tag_embedding = nn.Parameter(torch.empty(num_tags, embed_dim), requires_grad=False)
    
    def init_weights(self):
        if self.data_prefix is None:
            nn.init.kaiming_normal_(self.tag_embedding)
        else:
            import os.path as osp
            from tqdm import tqdm
            import numpy as np
            assert self.tag_map_file is not None, \
                'Require tag file'
            with open(self.tag_map_file, 'r') as F:
                lines = F.readlines()
            for i, tag in tqdm(enumerate(lines), desc='Loading pretrained features...'):
                tag = tag.strip()
                feat_file = osp.join(self.data_prefix, f'{tag}.npy')
                feat = np.load(feat_file)
                self.tag_embedding[i] = torch.tensor(feat)
        if not self.freeze:
            self.tag_embedding.requires_grad = True


@RECOMMENDERS.register_module()
class TagRecommender(nn.Module):
    """Tag recommender model framework.
    
    Args:
        tag_tower (dict): tag tower to get tag embeddings
    
    """

    def __init__(self,
                 tag_tower,
                 linker,
                 train_cfg=None):
        super().__init__()
        self.tag_tower = TagTower(**tag_tower)
        self.linker = builder.build_linker(linker)
        
        # aux_info is the list of tensor names beyond 'imgs' and 'label' which
        # will be used in train_step and val_step, data_batch should contain
        # these tensors
        self.aux_info = []
        if train_cfg is not None and 'aux_info' in train_cfg:
            self.aux_info = train_cfg['aux_info']
        
        self.init_weights()

    def init_weights(self):
        """Initialize the model network weights."""
        self.tag_tower.init_weights()
        self.linker.init_weights()

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        losses = dict()

        cls_score = self.linker(imgs, self.tag_tower.tag_embedding)
        loss_cls = self.linker.loss(cls_score, labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        cls_score = self.linker(imgs, self.tag_tower.tag_embedding)

        return cls_score.cpu().numpy()

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, imgs, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(imgs, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        imgs = data_batch['imgs']
        label = data_batch['label']

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self(imgs, label, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        imgs = data_batch['imgs']
        label = data_batch['label']

        aux_info = {}
        for item in self.aux_info:
            aux_info[item] = data_batch[item]

        losses = self(imgs, label, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs