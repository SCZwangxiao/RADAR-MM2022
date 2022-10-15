import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


@LOSSES.register_module
class DistributionBalancedBCELossWithLogits(BaseWeightedLoss):
    """Distribution-Balanced Binary Cross Entropy Loss with logits.
    from Distribution-Balanced Loss for Multi-label Classification in Long-Tailed Datasets.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """
    def __init__(self,
                 reduction='mean',
                 focal=dict(
                     focal=True,
                     balance_param=2.0,
                     gamma=2,
                 ),
                 map_param=dict(
                     alpha=0.1,
                     beta=10.0,
                     gamma=0.3
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.05
                 ),
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 freq_file='./class_freq.pkl',
                 loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)

        self.loss_weight = loss_weight
        self.reduction = reduction

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.balance_param = focal['balance_param']

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # Load class frequency
        self.__load_class_frequency__(freq_file)

        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias / self.neg_scale
        freq_inv = torch.ones(self.class_freq.shape) / self.class_freq
        self.register_buffer('init_bias', init_bias)
        self.register_buffer('freq_inv', freq_inv)
    
    def __load_class_frequency__(self, freq_file):
        import pickle
        with open(freq_file, 'rb') as F:
            class_freq = pickle.load(F)
        self.register_buffer(
            'class_freq', 
            torch.from_numpy(
                class_freq['class_freq']).float())
        self.register_buffer(
            'neg_class_freq', 
            torch.from_numpy(
                class_freq['neg_class_freq']).float())
        self.num_classes = self.class_freq.shape[0]
        self.train_num = self.class_freq[0] + self.neg_class_freq[0]

    def cls_criterion(self,
                      pred,
                      label,
                      weight=None,
                      reduction='mean',
                      avg_factor=None):
        if pred.dim() != label.dim():
            label, weight = _expand_binary_labels(label, weight, pred.size(-1))

        # weighted element-wise losses
        if weight is not None:
            weight = weight.float()

        loss = F.binary_cross_entropy_with_logits(
            pred, label.float(), weight, reduction='none')
        loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

        return loss

    def _forward(self,
                 cls_score,
                 label,
                 weight=None,
                 avg_factor=None,
                 reduction_override=None,
                 **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        if self.focal:
            logpt = - self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(logpt)
            loss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            loss = ((1 - pt) ** self.gamma) * loss
            loss = self.balance_param * loss
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        weight = self.rebalance_weight(label.float())

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None):
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale  + logits * labels
            weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight