# Copyright (c) 2021 Toyota Motor Europe
# Patent Pending. All rights reserved.
#
# Author: Michal Neoral, CMP FEE CTU Prague
# Contact: neoramic@fel.cvut.cz
#
# This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>

import torch
import torch.nn as nn

from mmdet.models.builder import build_backbone, build_neck
from raptor.mmdet_custom.models.builder import MOTION_BACKBONES, build_motion_backbone
from mmdet.models.detectors.base import BaseDetector


@MOTION_BACKBONES.register_module()
class MotionBackboneWithDetectoRS(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone=None,
                 motion_cost_volume=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(MotionBackboneWithDetectoRS, self).__init__()
        self.gpu_free_counter = 0

        self.motion_backbone_with_detectors = True

        self.additional_outputs_setting = None

        if motion_cost_volume is not None:
            self.motion_cost_volume = build_motion_backbone(motion_cost_volume)

        if backbone is not None:
            self.backbone = build_backbone(backbone)
            self.backbone.training = False
            self.backbone.eval()

        if neck is not None:
            self.neck = build_neck(neck)
            self.neck.training = False
            self.neck.eval()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.siamese = motion_cost_volume.get('siamese', False)
        if self.siamese:
            self.siamese_network = []
            for _ in range(neck['num_outs']):
                self.siamese_network.append(nn.Conv2d(
                    neck['out_channels'], # in channels
                    motion_cost_volume['siamese_outputs'], # out channels
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True))
        if self.siamese:
            self.siamese_module = nn.ModuleList(self.siamese_network)

    def set_additional_outputs_setting(self, additional_outputs_setting=None):
        self.additional_outputs_setting = additional_outputs_setting

        if self.with_motion_cost_volume:
            self.motion_cost_volume.set_additional_outputs_setting(additional_outputs_setting=additional_outputs_setting)

    def get_additional_outputs(self):
        if self.with_motion_cost_volume:
            return self.motion_cost_volume.get_additional_outputs()
        else:
            return None


    @property
    def with_motion_cost_volume(self):
        """bool: whether the detector has a motion backbone"""
        return hasattr(self, 'motion_cost_volume') and self.motion_cost_volume is not None

    @property
    def with_backbone(self):
        return hasattr(self, 'backbone') and self.backbone is not None

    def extract_feat(self, img, **kwargs):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        if self.siamese:
            for i in range(len(x)):
                x[i] = self.siamese_network[i](x[i])
        return x

    def forward_dummy(self, img, **kwargs):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()

        _, costs = self.motion_backbone(img=img, **kwargs)
        return self.extract_feat(costs)

    def forward_siamese(self, costs, **kwargs):
        b = int(costs.shape[1] // 2)
        c1 = costs[:, :b, :, :]
        c2 = costs[:, b:, :, :]
        x1 = self.extract_feat(c1, **kwargs)
        x2 = self.extract_feat(c2, **kwargs)
        return [torch.cat([x1[i], x2[i]], dim=1) for i in range(len(x1))]

    def forward_train(self,
                      img,
                      img_metas,
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

        # if self.multiple_images:
        #     self.backbone.training = False
        #     self.backbone.eval()
        #     self.neck.training = False
        #     self.neck.eval()
        #     return self.forward_multiple_images(img=img, features=None, img_metas=img_metas, simple_test=False, pseudo_training=True, **kwargs)

        _, costs = self.motion_cost_volume(img=img, features=None, img_metas=img_metas, simple_test=False, pseudo_training=True, **kwargs)
        if self.siamese is True:
            return self.forward_siamese(costs, **kwargs)
        else:
            return self.extract_feat(costs, **kwargs)


    # def forward_multiple_images(self, img=None, features=None, img_metas=None, simple_test=False, pseudo_training=True, **kwargs):
    #     x_list = []
    #     for d_img in ['tp', 'tm']:
    #         for t_img in range(1,20):
    #             key_img = f'img_{d_img}_{t_img}'
    #             if key_img not in kwargs:
    #                 continue
    #             key_calib = f'calib_K_{d_img}_{t_img}'
    #             key_pos = f'calib_position_{d_img}_{t_img}'
    #             kwargs['img2'] = kwargs[key_img]
    #             kwargs['calib_K_2'] = kwargs[key_calib]
    #             if not simple_test:
    #                 kwargs['calib_position_2'] = kwargs[key_pos]
    #             _, costs = self.motion_cost_volume(img=img,
    #                                                features=features,
    #                                                simple_test=simple_test,
    #                                                img_metas=img_metas,
    #                                                pseudo_training=pseudo_training,
    #                                                output_cost_only=True, **kwargs)
    #             x = self.extract_feat(costs, **kwargs)
    #             x_list.append(x)
    #     assert len(x_list) > 0
    #     return [
    #         self.multiple_images_network[i](torch.cat(x2cat, dim=1))
    #         for i, x2cat in enumerate(zip(*x_list))
    #     ]

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False,
                                **kwargs):
        raise NotImplementedError('Not implemented')

    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""

        # if self.multiple_images:
        #     return self.forward_multiple_images(img=img, features=None, img_metas=img_metas, simple_test=True, **kwargs)

        _, costs = self.motion_cost_volume(img=img, features=None, img_metas=img_metas, simple_test=True, **kwargs)
        if self.siamese is True:
            return self.forward_siamese(costs, **kwargs)
        else:
            return self.extract_feat(costs, **kwargs)


    def extract_multiple_image_features(self, feature_list):
        return None

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        raise NotImplementedError('Motion is not implemented')
