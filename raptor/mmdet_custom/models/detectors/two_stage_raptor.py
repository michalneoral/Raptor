# Copyright (c) 2021 Toyota Motor Europe
# Patent Pending. All rights reserved.
#
# Author: Michal Neoral, CMP FEE CTU Prague
# Contact: neoramic@fel.cvut.cz
#
# This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>


import torch

from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from raptor.mmdet_custom.models.builder import build_motion_backbone
from mmdet.models.detectors.base import BaseDetector


@DETECTORS.register_module()
class TwoStageDetectorRaptor(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone=None,
                 motion_backbone=None,
                 motion_backbone_before_backbone=None,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetectorRaptor, self).__init__()

        self.bw_flow_output = False
        self.fw_flow_output = False

        if backbone is not None:
            self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if motion_backbone is not None:
            motion_backbone_ = motion_backbone.copy()
            motion_backbone_.update(train_cfg=train_cfg, test_cfg=test_cfg)
            self.motion_backbone = build_motion_backbone(motion_backbone_)

        if motion_backbone_before_backbone is not None:
            motion_backbone_ = motion_backbone_before_backbone.copy()
            motion_backbone_.update(train_cfg=train_cfg, test_cfg=test_cfg)
            self.motion_backbone_before_backbone = build_motion_backbone(motion_backbone_)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    @property
    def with_motion_backbone(self):
        """bool: whether the detector has a motion backbone"""
        return hasattr(self, 'motion_backbone') and self.motion_backbone is not None

    @property
    def with_motion_backbone_before_backbone(self):
        """bool: whether the detector has a motion backbone"""
        return hasattr(self, 'motion_backbone_before_backbone') and self.motion_backbone_before_backbone is not None

    @property
    def with_backbone(self):
        return hasattr(self, 'backbone') and self.backbone is not None

    def set_additional_outputs(self, additional_outputs_setting=None):
        self.additional_outputs_setting = additional_outputs_setting

        if self.with_motion_backbone:
            self.motion_backbone.set_additional_outputs_setting(additional_outputs_setting=additional_outputs_setting)

    def get_additional_outputs(self):
        if self.with_motion_backbone:
            return self.motion_backbone.get_additional_outputs()
        else:
            return None, None

    def extract_feat(self, img, **kwargs):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img, **kwargs):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        if self.with_motion_backbone:
            x, costs = self.motion_backbone(img=img, features=x, **kwargs)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet_custom/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if self.with_motion_backbone and self.motion_backbone.motion_backbone_with_detectors:
            if self.motion_backbone.multiple_images:
                self.neck.training = False
                self.neck.eval()
                self.backbone.training = False
                self.backbone.eval()
            motion_features = self.motion_backbone(img, img_metas=img_metas, **kwargs)
            semantic_features = self.extract_feat(img, **kwargs)

            assert len(motion_features) == len(semantic_features)
            x = [
                torch.cat([motion_features[i], semantic_features[i]], dim=1)
                for i in range(len(motion_features))
            ]

        elif self.with_motion_backbone_before_backbone:
            _, costs = self.motion_backbone_before_backbone(img=img, features=None, img_metas=img_metas, pseudo_training=True, **kwargs)
            x = self.extract_feat(costs, **kwargs)
        elif self.with_backbone:
            x = self.extract_feat(img, **kwargs)
            if self.with_motion_backbone:
                x, costs = self.motion_backbone(img=img, features=x, img_metas=img_metas, pseudo_training=True, **kwargs)
        elif self.with_motion_backbone:
            x, costs = self.motion_backbone(img=img, features=None, img_metas=img_metas, pseudo_training=True, **kwargs)
        else:
            raise NotImplementedError('No option for this config')

        losses = {}

        # RPN forward and loss
        print([(len(gt_bboxes[i]), len(gt_labels[i]), len(gt_masks[i])) for i in range(len(gt_bboxes))])
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        print([len(proposal_list[i]) for i in range(len(proposal_list))])
        # with SetTrace(self.CMT.monitor):
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        if torch.isnan(rpn_losses['loss_rpn_cls'][0]):
            print('nan')

        self.CMT.set_active()
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False,
                                **kwargs):
        """Async test without augmentation.
           NOT IMPLEMENTED
        """
        raise NotImplementedError('Async test is not implemented for Raptor')

    def format_inputs_for_test(self, img, kwargs):
        if not isinstance(img, list):
            if 'img2' in kwargs:
                if isinstance(kwargs['img2'], list):
                    kwargs['img2'] = kwargs['img2'][0]
                    kwargs['calib_K'][0] = kwargs['calib_K'][0][0]
                    kwargs['calib_K_2'][0] = kwargs['calib_K_2'][0][0]
                    kwargs['calib_baseline'][0] = kwargs['calib_baseline'][0][0]
                    if 'img0' in kwargs:
                        kwargs['img0'] = kwargs['img0'][0]
                        kwargs['calib_K_0'][0] = kwargs['calib_K_0'][0][0]
            elif 'img_tm_1' in kwargs:
                if isinstance(kwargs['img_tm_1'], list):
                    kwargs['calib_K'][0] = kwargs['calib_K'][0][0]
                    kwargs['calib_baseline'][0] = kwargs['calib_baseline'][0][0]
                    for d in ['tm', 'tp']:
                        for i in range(5):
                            if f'img_{d}_{i}' in kwargs:
                                kwargs[f'img_{d}_{i}'] = kwargs[f'img_{d}_{i}'][0]
                                kwargs[f'calib_K_{d}_{i}'][0] = kwargs[f'calib_K_{d}_{i}'][0][0]
        return img, kwargs

    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        print('keys: \n', kwargs.keys())

        img, kwargs = self.format_inputs_for_test(img, kwargs)

        if self.with_motion_backbone and self.motion_backbone.motion_backbone_with_detectors:
            motion_features = self.motion_backbone.simple_test(img, img_metas=img_metas, **kwargs)
            semantic_features = self.extract_feat(img, **kwargs)
            assert len(motion_features) == len(semantic_features)
            x = [
                torch.cat([motion_features[i], semantic_features[i]], dim=1)
                for i in range(len(motion_features))
            ]

        elif self.with_motion_backbone_before_backbone:
            _, costs = self.motion_backbone_before_backbone(img=img, features=None, simple_test=True, img_metas=img_metas, **kwargs)
            x = self.extract_feat(costs, **kwargs)
        elif self.with_backbone:
            x = self.extract_feat(img, **kwargs)
            if self.with_motion_backbone:
                x, costs = self.motion_backbone(img=img, features=x, simple_test=True, img_metas=img_metas, **kwargs)
        elif self.with_motion_backbone:
            x, costs = self.motion_backbone(img=img, features=None, simple_test=True, img_metas=img_metas, **kwargs)
        else:
            raise NotImplementedError('No option for this config')

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].

        NOT IMPLEMENTED
        """
        raise NotImplementedError('Augmentation test is not implemented for Raptor')
