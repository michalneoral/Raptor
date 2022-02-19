# Copyright (c) 2021 Toyota Motor Europe
# Patent Pending. All rights reserved.
#
# Author: Michal Neoral, CMP FEE CTU Prague
# Contact: neoramic@fel.cvut.cz
#
# This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
from kornia.geometry.transform import crop_and_resize as kornia_crop_resize

from raptor.mmdet_custom.models.motion_backbones.external_code_fragments.vcnplus.vcnplus_used_code import affine_exp, depth_change_net, compute_optical_expansion, compute_mcv

from RAFT.core.raft import RAFT
from raptor.mmdet_custom.models.builder import MOTION_BACKBONES


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__.update(kwargs)


@MOTION_BACKBONES.register_module()
class RAFTplus(nn.Module):
    """
    RAFTplus - computation of motion cost volume using optical flow, monodepth and optical expansion
    """
    def __init__(self, max_disp, fac, exp_unc, samples_per_gpu, img_scale, **kwargs):
        super(RAFTplus, self).__init__()

        self.additional_outputs_setting = None
        self.additional_outputs = None
        self.computed_networks_outputs = {}

        self.iteration_number = 0

        self.motion_backbone_with_detectors = False
        self.img_norm_cfg = kwargs.get('img_norm_cfg', None)
        self.stage = kwargs.get('stage', 2)
        self.switched_camera_positions = kwargs.get('switched_camera_positions', False)
        self.dual_network = kwargs.get('dual_network', False) # cost volume will be pruduced in BW FW manner (it produces 28 channel cost volume instead of 14)
        self.add_padding_channel = kwargs.get('add_padding_channel', False)
        self.use_opencv = kwargs.get('use_opencv', False)

        self.pseudo_training = True
        self.debug = False

        self.md = [int(4*(max_disp/256)), 4,4,4,4]
        self.fac = int(fac)

        self.orig_size = [samples_per_gpu, img_scale[0], img_scale[1]] # B x W x H
        max_h = int(np.ceil(img_scale[1] / 64) * 64 )
        max_w = int(np.ceil(img_scale[0] / 64) * 64 )
        self.new_size = [samples_per_gpu, max_w, max_h]

        # RAFT init
        raft_params = AttrDict(**kwargs)

        self.raft = RAFT(raft_params)
        self.raft.requires_grad_(False)
        self.raft.eval()
        self.raft_iters = kwargs.get('iters', 24)

        # OpticalExpansion init
        # affine-exp
        affine_exp(self)

        # depth change net
        depth_change_net(self, exp_unc, input_features=512)

        # MIDAS init
        model_type = "MiDaS"
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type in {"DPT_Large", "DPT_Hybrid"}:
            self.midas_transform = midas_transforms.dpt_transform
        else:
            self.midas_transform = midas_transforms.small_transform

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def set_additional_outputs_setting(self, additional_outputs_setting=None):
        self.additional_outputs_setting = additional_outputs_setting
        if additional_outputs_setting is not None:
            self.additional_outputs = {}

    def get_additional_outputs(self):
        if self.additional_outputs is None:
            return None
        return {
            key: self.additional_outputs[key]
            for key in self.additional_outputs_setting
            if key in self.additional_outputs
        }

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def init_weights(self, pretrained=None):
        pass

    def compute_camera_motion(self, calib_position, calib_position_2):
        if self.switched_camera_positions:
            ext1 = torch.stack(calib_position, dim=0)
            ext0 = torch.stack(calib_position_2, dim=0)
        else:
            ext0 = torch.stack(calib_position, dim=0)
            ext1 = torch.stack(calib_position_2, dim=0)

        B = len(calib_position)

        RT01 = [self.compute_camera_motion_numpy(ext0[i], ext1[i]) for i in range(B)]
        RT01 = np.stack(RT01, axis=0)
        RT01 = torch.from_numpy(RT01).float().cuda(ext0.device)
        return RT01

    def compute_camera_motion_numpy(self, ext0, ext1):
        ext0 = ext0.detach().cpu().numpy()
        ext1 = ext1.detach().cpu().numpy()
        camT = np.eye(4)
        camT[1, 1] = -1
        camT[2, 2] = -1
        RT01 = camT.dot(np.linalg.inv(ext0)).dot(ext1).dot(camT)
        RT01 = np.concatenate((cv2.Rodrigues(RT01[:3, :3])[0][:, 0], RT01[:3, -1]))
        return RT01

    def augment_inputs(self):
        for _ in range(100):
            print('MISSING DATA AUGMENTATION')

    def prepare_camera_intr(self, calib_K, calib_baseline, intr_aug=None):
        bl = torch.cat(calib_baseline, dim=0)
        ap_zero = torch.zeros_like(bl)
        ap_one = torch.ones_like(bl)
        intr_pr = torch.stack(calib_K, dim=0)
        focal = intr_pr[:, 0, 0]
        cx = intr_pr[:, 0, 2]
        cy = intr_pr[:, 1, 2]

        if intr_aug is not None:
            intr = intr_aug
        else:
            intr = [focal, cx, cy, bl, ap_one, ap_zero, ap_zero, ap_one, ap_zero, ap_zero]

        if self.new_size is not None:
            self.prepare_camera_according_new_size(intr, ap_one, focal)
        return intr

    def prepare_camera_according_new_size(self, intr, ap_one, focal):
        maxh = self.new_size[2]
        maxw = self.new_size[1]
        origh = self.orig_size[2]
        origw = self.orig_size[1]
        intr.append(ap_one * origw / maxw)
        intr.append(ap_one * origh / maxh)
        intr.append(focal)

    def remove_padding_and_resize(self, img, padding_mask=None):

        batch_size = img.shape[0]
        old_h = self.orig_size[2]
        old_w = self.orig_size[1]
        new_h = self.new_size[2]
        new_w = self.new_size[1]
        cur_h = img.shape[2]
        cur_w = img.shape[3]

        pad_h = (cur_h - old_h)
        pad_w = (cur_w - old_w)
        self.pad_h = pad_h
        self.pad_w = pad_w

        if padding_mask is not None:
            pm = torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
            pm = F.pad(pm, (0, self.pad_w, 0, self.pad_h))

            img[pm==0] = 0

        # boxes (torch.Tensor) – a tensor containing the coordinates of the bounding boxes to be extracted.
        # The tensor must have the shape of Bx4x2, where each box is defined in the following (clockwise) order:
        # top-left, top-right, bottom-right and bottom-left.
        # The coordinates must be in the x, y order. The coordinates would compose a rectangle with a shape of (N1, N2).
        bbox_x = np.array([ 0, cur_w - pad_w - 1, cur_w - pad_w - 1, 0 ])
        bbox_y = np.array([ 0, 0, cur_h - pad_h - 1, cur_h - pad_h - 1 ])
        bbox = np.stack([bbox_x, bbox_y], axis=1)
        bbox = np.stack([bbox] * batch_size, axis=0)
        bbox = torch.from_numpy(bbox).float().cuda(img.device)

        img_croped_resized = kornia_crop_resize(img, bbox, (new_h, new_w), align_corners=True)
        return img_croped_resized.detach()

    def prepare_outputs(self, costs, features, add_padding_channel=False, padding_mask=None, output_cost_only=False):
        h = self.orig_size[2]
        w = self.orig_size[1]
        # costs = kornia_resize(costs, size=(h,w), align_corners=True)
        costs = F.interpolate(costs, size=(h, w), mode='bilinear', align_corners=True)

        if padding_mask is not None:
            n = costs.shape[1]
            pc = torch.unsqueeze(padding_mask, 1)

            pc = pc.repeat((1,n,1,1))
            costs[pc==0] = 0

        costs = F.pad(costs, (0, self.pad_w, 0, self.pad_h))

        if add_padding_channel:
            b, _, _, _ = costs.shape
            add_c = torch.ones(size=(b, 1, h, w), dtype=costs.dtype, device=costs.device)
            add_c = F.pad(add_c, (0, self.pad_w, 0, self.pad_h), value=-1)
            costs = torch.cat([costs, add_c], dim=1)

        if features is not None and not output_cost_only:
            features = list(features)
            for i in range(len(features)):
                if costs.shape[2:] == features[i].shape[2:]:
                    c = costs
                else:
                    c = F.interpolate(costs, (features[i].shape[2:]))
                    #with torch.set_grad_enabled(self.training):
                features[i] = torch.cat([features[i], c], dim=1)
        elif output_cost_only:
            return None, costs
        else:
            features = []
            for i in range(5):
                size_coef = 2**(i+2)
                c = F.interpolate(costs, (costs.shape[2]//size_coef, costs.shape[3]//size_coef))
                features.append(c)
        return features, costs

    def prepare_images(self, img, img2, padding_mask=None, padding_mask_2=None):
        # denormalisation
        mean_L = np.asarray([[0.33, 0.33, 0.33]]).mean(0)[np.newaxis, :, np.newaxis, np.newaxis]
        mean_R = np.asarray([[0.33, 0.33, 0.33]]).mean(0)[np.newaxis, :, np.newaxis, np.newaxis]
        mean_L = torch.from_numpy(mean_L).float().cuda(img.device)
        mean_R = torch.from_numpy(mean_R).float().cuda(img.device)

        img_norm_cfg = self.img_norm_cfg
        mean = img_norm_cfg.mean
        std = img_norm_cfg.std
        mean = torch.from_numpy(np.array(mean)).float().reshape(1, -1, 1, 1)
        std = torch.from_numpy(np.array(std)).float().reshape(1, -1, 1, 1)
        mean = mean.cuda(img.device)
        std = std.cuda(img.device)
        imgoL = (img * std + mean) / 255.
        imgoR = (img2 * std + mean) / 255.

        # resize for MIDAS and VCN
        imgoL = self.remove_padding_and_resize(imgoL, padding_mask=padding_mask)
        imgoR = self.remove_padding_and_resize(imgoR, padding_mask=padding_mask_2)

        imgRAFT = torch.cat([imgoL, imgoR], dim=0)
        imgRAFT = imgRAFT * 2. - 1.

        # preparation of Left/Right inputs for MIDAS and VCN
        imgL = imgoL.detach().clone()[:, [2, 1, 0], :, :] - mean_L  # [B x BGR x H x W] - format for VCN
        imgR = imgoR.detach().clone()[:, [2, 1, 0], :, :] - mean_R  # [B x BGR x H x W] - format for VCN

        imgoL = imgoL.transpose(1, 3).transpose(1, 2)  # [B x H x W x RGB] - format for ?
        imgoR = imgoR.transpose(1, 3).transpose(1, 2)  # [B x H x W x RGB] - format for ?

        imgVCN = torch.cat([imgL, imgR], dim=0)  # input to VCN
        return imgVCN, imgRAFT, imgoL, imgoR


    def prepare_inputs(self, img, img2, calib_position, calib_position_2, calib_K, calib_K_2, calib_baseline, disp_input, intr_aug=None, padding_mask=None, padding_mask_2=None):
        if calib_position and calib_position_2 is not None:
            RT01 = self.compute_camera_motion(calib_position, calib_position_2)
        else:
            RT01 = None
        intr = self.prepare_camera_intr(calib_K, calib_baseline, intr_aug=intr_aug)
        imgVCN, imgRAFT, imgoL, imgoR = self.prepare_images(img, img2, padding_mask=padding_mask, padding_mask_2=padding_mask_2)
        occp = torch.zeros([len(calib_K), 4], dtype=torch.float).cuda(img.device)
        disk_aux = [None, None, None, intr, imgoL, imgoR, occp, RT01, self.stage]
        return imgVCN, imgRAFT, disk_aux, disp_input


    def forward(self, features, img, img2, img0=None,
                calib_position=None, calib_position_2=None, calib_K=None, calib_K_2=None, calib_baseline=None,
                disp_input=None, simple_test=False,
                calib_position_0=None, calib_K_0=None, img_metas=None,
                intr_aug=None, padding_mask=None, padding_mask_2=None, padding_mask_0=None, output_cost_only=False,
                **kwargs):

        self.simple_test = simple_test
        self.img_metas = img_metas
        bs = len(img_metas)
        ori_shape = img_metas[0]['ori_shape']
        pad_shape = img_metas[0]['pad_shape']

        self.new_size = [bs, pad_shape[1], pad_shape[0]]
        self.orig_size = [bs, ori_shape[1], ori_shape[0]]

        self.pseudo_training = bool(self.training)
        self.eval()
        torch.set_grad_enabled(False)

        im, imRAFT, extended_input_data, disp_input = self.prepare_inputs(img, img2, calib_position, calib_position_2, calib_K, calib_K_2, calib_baseline, disp_input, intr_aug=intr_aug, padding_mask=padding_mask, padding_mask_2=padding_mask_2)

        outputs_raftplus = self.raftplus_forward(im, imRAFT, extended_input_data, disp_input=disp_input, additional_outputs_setting=self.additional_outputs_setting)
        if self.additional_outputs_setting is not None:
            outputs_raftplus, additional_outputs = outputs_raftplus[0], outputs_raftplus[1]
            self.additional_outputs['flow_t_t+1'] = additional_outputs['flow_fw']
            self.additional_outputs['flow_t+1_t'] = additional_outputs['flow_bw']
        outputs_raftplus = outputs_raftplus.detach()

        if self.dual_network:
            im, imRAFT, extended_input_data, disp_input = self.prepare_inputs(img, img0, calib_position, calib_position_0, calib_K, calib_K_0, calib_baseline, disp_input)
            outputsBW_raftplus = self.raftplus_forward(im, imRAFT, extended_input_data, disp_input=disp_input, additional_outputs_setting=self.additional_outputs_setting)
            if self.additional_outputs_setting is not None:
                outputsBW_raftplus, additionalBW_outputs = outputsBW_raftplus[0], outputsBW_raftplus[1]
                self.additional_outputs['flow_t_t-1'] = additionalBW_outputs['flow_fw']
                self.additional_outputs['flow_t-1_t'] = additionalBW_outputs['flow_bw']
            outputsBW_raftplus = outputsBW_raftplus.detach()
            outputs_raftplus = torch.cat([outputs_raftplus, outputsBW_raftplus], dim=1)

        self.requires_grad_(False)

        if self.pseudo_training:
            self.train()
            torch.set_grad_enabled(True)
        outputs, costs = self.prepare_outputs(outputs_raftplus,
                                              features,
                                              add_padding_channel=self.add_padding_channel,
                                              padding_mask=padding_mask,
                                              output_cost_only=output_cost_only)

        self.requires_grad_(False)
        return outputs, costs

    def raftplus_forward(self, im, imRAFT, extended_input_data, disp_input=None, additional_outputs_setting=None):
        cuda_device_name = im.device

        self.flow_raft_fw_full = None
        self.flow_raft_bw_full = None
        bs = im.shape[0] // 2

        image1 = imRAFT[:bs, :, :, :]
        image2 = imRAFT[bs:, :, :, :]
        self.raft.eval()
        flow_raft_fw_full, features_raft = self.raft(image1, image2, iters=self.raft_iters, test_mode=True, normalise_input=False, return_features=True, return_coords=False)
        ### self.computed_networks_outputs['flow_raft_fw_full'] = flow_raft_fw_full

        flow_raft = F.interpolate(flow_raft_fw_full, size=(im.shape[2] // 4, im.shape[3] // 4), mode='bilinear', align_corners=True) / 4

        ### compute bw/fw consistency
        self.raft.eval()
        flow_raft_bw_full, _ = self.raft(image2, image1, iters=self.raft_iters, test_mode=True, normalise_input=False, return_features=True, return_coords=False)
        # self.computed_networks_outputs['flow_raft_bw_full'] = flow_raft_bw_full

        ### optical expansion
        features_raft = F.interpolate(features_raft, size=(im.shape[2] // 4, im.shape[3] // 4), mode='bilinear', align_corners=True)
        b, _, h, w = flow_raft.shape
        dchange2, dc_unc, iexp2 = compute_optical_expansion(self, im, flow_raft, features_raft, extended_input_data)

        with torch.no_grad():
            self.midas.eval()
            input_im = (extended_input_data[4].permute(0, 3, 1, 2) -
                        torch.Tensor([0.485, 0.456, 0.406]).cuda(cuda_device_name)[np.newaxis, :, np.newaxis, np.newaxis]) / \
                        torch.Tensor([0.229, 0.224, 0.225]).cuda(cuda_device_name)[np.newaxis, :, np.newaxis, np.newaxis]
            wsize = int((input_im.shape[3] * 448. / input_im.shape[2]) // 32 * 32)
            input_im = F.interpolate(input_im, (448, wsize), mode='bilinear')
            input_im_for_midas = input_im
            dispo = self.midas.forward(input_im_for_midas)[None].clamp(1e-6, np.inf)

        flow = flow_raft_fw_full

        ### compute motion cost volume
        costs = compute_mcv(im,
                dispo,
                dchange2, dc_unc,
                flow, flow_raft_fw_full, flow_raft_bw_full,
                cuda_device_name, bs, extended_input_data, self.training, self.pseudo_training, self.debug, self.use_opencv)

        if self.additional_outputs_setting is not None:
            return costs, {'flow_fw': flow_raft_fw_full.detach(), 'flow_bw': flow_raft_bw_full.detach()}
        return costs
