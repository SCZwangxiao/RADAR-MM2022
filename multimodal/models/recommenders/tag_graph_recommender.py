import warnings
from collections import OrderedDict

import torch
from torch import nn
import torch.distributed as dist

from .. import builder
from ..builder import RECOMMENDERS


@RECOMMENDERS.register_module()
class TagGraphRecommender(nn.Module):
    """Tag recommender model framework.
    
    Args:
        tag_tower (dict): tag tower to get tag embeddings
    
    """

    def __init__(self,
                 gnn,
                 linker,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.gnn = builder.build_gnn(gnn)
        self.linker = builder.build_linker(linker)
        
        self.init_weights()

    def init_weights(self):
        """Initialize the model network weights."""
        self.gnn.init_weights()
        self.linker.init_weights()

    def forward_train(self, input_nodes, pair_graph, mfgs):
        """Defines the computation performed at every call when training."""
        losses = dict()

        self.gnn(input_nodes, pair_graph, mfgs)
        cls_score, labels = self.linker(input_nodes, pair_graph, mfgs)
        loss = self.linker.loss(cls_score, labels, pair_graph, mfgs)
        losses.update(loss)

        return losses

    def forward_test(self, input_nodes, pair_graph, mfgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        self.gnn.infer(input_nodes, pair_graph, mfgs)
        cls_score = self.linker.infer(input_nodes, pair_graph, mfgs)

        return cls_score.cpu().numpy()
    
    def _to_device(self, input_nodes, pair_graph, mfgs):
        device = input_nodes['tag'].device
        pair_graph = pair_graph.to(device)
        mfgs = [mfg.to(device) for mfg in mfgs]
        return input_nodes, pair_graph, mfgs

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

    def forward(self, input_nodes, pair_graph, mfgs, return_loss=True):
        """Define the computation performed at every call."""
        input_nodes, pair_graph, mfgs = self._to_device(input_nodes, pair_graph, mfgs)
        if return_loss:
            return self.forward_train(input_nodes, pair_graph, mfgs)

        return self.forward_test(input_nodes, pair_graph, mfgs)
    
    def __process_batch__(self, data_batch):
        input_nodes = None
        if 'input_nodes' in data_batch:
            input_nodes = data_batch['input_nodes']
        pair_graph = data_batch['pair_graph']
        mfgs = data_batch['mfgs']
        batch_size = pair_graph.num_nodes('video')
        return input_nodes, pair_graph, mfgs, batch_size

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
        input_nodes, pair_graph, mfgs, batch_size = self.__process_batch__(data_batch)

        losses = self(input_nodes, pair_graph, mfgs, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=batch_size)

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        input_nodes, pair_graph, mfgs, batch_size = self.__process_batch__(data_batch)

        losses = self(input_nodes, pair_graph, mfgs, return_loss=False)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=batch_size)

        return outputs