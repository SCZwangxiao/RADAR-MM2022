import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base_multimodal import BaseMultiModalRecognizer


@RECOGNIZERS.register_module()
class MultiModalRecognizer2D(BaseMultiModalRecognizer):
    """Basic multimodal model framework."""

    def forward_train(self, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head

        losses = dict()

        if self.vision_backbone:
            kwargs['imgs'] = self.extract_feat(kwargs['imgs'], 'vision')
        if self.language_backbone:
            kwargs['sents'] = self.extract_feat(kwargs['sents'], 'language')
        if self.audio_backbone:
            kwargs['audios'] = self.extract_feat(kwargs['audios'], 'audio')
        
        if self.vision_encoder:
            kwargs['imgs'] = self.vision_encoder(kwargs['imgs'])
        if self.language_encoder:
            kwargs['sents'] = self.language_encoder(kwargs['sents'])
        if self.audio_encoder:
            kwargs['audios'] = self.audio_encoder(kwargs['audios'])

        """
        if self.vision_backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x, labels.squeeze())
            x = x.squeeze(2)
            num_segs = 1
            losses.update(loss_aux)
        """

        mm_feats = self.reactor(**kwargs)

        cls_score = self.cls_head(mm_feats)
        gt_labels = labels.squeeze()
        kwargs.pop('imgs', None)
        kwargs.pop('sents', None)
        kwargs.pop('aduios', None)
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, **kwargs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        if self.vision_backbone:
            kwargs['imgs'] = self.extract_feat(kwargs['imgs'], 'vision')
        if self.language_backbone:
            kwargs['sents'] = self.extract_feat(kwargs['sents'], 'language')
        if self.audio_backbone:
            kwargs['audios'] = self.extract_feat(kwargs['audios'], 'audio')
        
        if self.vision_encoder:
            kwargs['imgs'] = self.vision_encoder(kwargs['imgs'])
        if self.language_encoder:
            kwargs['sents'] = self.language_encoder(kwargs['sents'])
        if self.audio_encoder:
            kwargs['audios'] = self.audio_encoder(kwargs['audios'])
        """
        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1
        
        if self.feature_extraction:
            # perform spatial pooling
            avg_pool = nn.AdaptiveAvgPool2d(1)
            x = avg_pool(x)
            # squeeze dimensions
            x = x.reshape((batches, num_segs, -1))
            # temporal average pooling
            x = x.mean(axis=1)
            return x
        """

        mm_feats = self.reactor(**kwargs)

        cls_score = self.cls_head(mm_feats)
        """
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        """
        return cls_score

    def _do_fcn_test(self, imgs):
        raise NotImplementedError
        # [N, num_crops * num_segs, C, H, W] ->
        # [N * num_crops * num_segs, C, H, W]
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = self.test_cfg.get('num_segs', self.backbone.num_segments)

        if self.test_cfg.get('flip', False):
            imgs = torch.flip(imgs, [-1])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
        else:
            x = x.reshape((-1, num_segs) +
                          x.shape[1:]).transpose(1, 2).contiguous()

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        cls_score = self.cls_head(x, fcn_test=True)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

    def forward_test(self, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        if self.test_cfg.get('fcn_test', False):
            # If specified, spatially fully-convolutional testing is performed
            assert not self.feature_extraction
            assert self.with_cls_head
            return self._do_fcn_test(**kwargs).cpu().numpy()
        return self._do_test(**kwargs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        raise NotImplementedError
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        outs = self.cls_head(x, num_segs)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, **kwargs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(**kwargs)
