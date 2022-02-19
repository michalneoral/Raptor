#

import torch
import torch.nn.functional as F
import numpy as np
import kornia

from raptor.mmdet_custom.models.motion_backbones.external_code_fragments.vcnplus.vcnplus_submodule import bfmodule, conv, compute_geo_costs, get_skew_mat, get_intrinsics, F_ngransac
from raptor.mmdet_custom.models.motion_backbones.external_code_fragments.vcnplus.vcnplus_submodule import get_grid


def affine(pref, flow, pw=1):
    b, _, lh, lw = flow.shape
    ptar = pref + flow
    pw = 1
    pref = F.unfold(pref, (pw * 2 + 1, pw * 2 + 1), padding=(pw)).view(b, 2, (pw * 2 + 1) ** 2, lh, lw) - pref[:, :, np.newaxis]
    ptar = F.unfold(ptar, (pw * 2 + 1, pw * 2 + 1), padding=(pw)).view(b, 2, (pw * 2 + 1) ** 2, lh, lw) - ptar[:, :, np.newaxis]  # b, 2,9,h,w
    pref = pref.permute(0, 3, 4, 1, 2).reshape(b * lh * lw, 2, (pw * 2 + 1) ** 2)
    ptar = ptar.permute(0, 3, 4, 1, 2).reshape(b * lh * lw, 2, (pw * 2 + 1) ** 2)

    prefprefT = pref.matmul(pref.permute(0, 2, 1))
    ppdet = prefprefT[:, 0, 0] * prefprefT[:, 1, 1] - prefprefT[:, 1, 0] * prefprefT[:, 0, 1]
    ppinv = torch.cat((prefprefT[:, 1, 1:], -prefprefT[:, 0, 1:], -prefprefT[:, 1:, 0], prefprefT[:, 0:1, 0]), 1).view(-1, 2, 2) / ppdet.clamp(1e-10, np.inf)[:, np.newaxis,
                                                                                                                                   np.newaxis]

    Affine = ptar.matmul(pref.permute(0, 2, 1)).matmul(ppinv)
    Error = (Affine.matmul(pref) - ptar).norm(2, 1).mean(1).view(b, 1, lh, lw)

    Avol = (Affine[:, 0, 0] * Affine[:, 1, 1] - Affine[:, 1, 0] * Affine[:, 0, 1]).view(b, 1, lh, lw).abs().clamp(1e-10, np.inf)
    exp = Avol.sqrt()
    mask = (exp > 0.5) & (exp < 2) & (Error < 0.1)
    mask = mask[:, 0]

    exp = exp.clamp(0.5, 2)
    exp[Error > 0.1] = 1
    return exp, Error, mask

def compute_optical_expansion(self, im, flow2, c12, disc_aux):
    if not self.training or disc_aux[-1]:
        b, _, h, w = flow2.shape
        exp2, err2, _ = affine(get_grid(b, h, w, cuda_device_name=flow2.device)[:, 0].permute(0, 3, 1, 2).repeat(b, 1, 1, 1).clone(), flow2.detach(), pw=1)
        x = torch.cat((
            self.f3d2v2(-exp2.log()),
            self.f3d2v3(err2),
        ), 1)
        dchange2 = -exp2.log() + 1. / 200 * self.f3d2(x)[0]

        # depth change net
        iexp2 = F.interpolate(dchange2.clone(), [im.size()[2], im.size()[3]], mode='bilinear', align_corners=True)
        x = torch.cat((self.dcnetv1(c12.detach()),
                       self.dcnetv2(dchange2.detach()),
                       self.dcnetv3(-exp2.log()),
                       self.dcnetv4(err2),
                       ), 1)
        dcneto = 1. / 200 * self.dcnet(x)[0]
        dchange2 = dchange2.detach() + dcneto[:, :1]
        dchange2 = F.interpolate(dchange2, [im.size()[2], im.size()[3]], mode='bilinear', align_corners=True)

        if dcneto.shape[1] > 1:
            dc_unc = dcneto[:, 1:2]
        else:
            dc_unc = torch.zeros_like(dcneto)
        dc_unc = F.interpolate(dc_unc, [im.size()[2], im.size()[3]], mode='bilinear', align_corners=True)[:, 0]

        return dchange2, dc_unc, iexp2

def affine_exp(self):
    # affine-exp
    self.f3d2v1 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)  #
    self.f3d2v2 = conv(1, 32, kernel_size=3, stride=1, padding=1, dilation=1)  #
    self.f3d2v3 = conv(1, 32, kernel_size=3, stride=1, padding=1, dilation=1)  #
    self.f3d2v4 = conv(1, 32, kernel_size=3, stride=1, padding=1, dilation=1)  #
    self.f3d2v5 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)  #
    self.f3d2v6 = conv(12 * 81, 32, kernel_size=3, stride=1, padding=1, dilation=1)  #
    self.f3d2 = bfmodule(128 - 64, 1)


def depth_change_net(self, exp_unc, input_features=64):
    # depth change net, raft change 64 -> 96
    self.dcnetv1 = conv(input_features, 32, kernel_size=3, stride=1, padding=1, dilation=1)  #
    self.dcnetv2 = conv(1, 32, kernel_size=3, stride=1, padding=1, dilation=1)  #
    self.dcnetv3 = conv(1, 32, kernel_size=3, stride=1, padding=1, dilation=1)  #
    self.dcnetv4 = conv(1, 32, kernel_size=3, stride=1, padding=1, dilation=1)  #
    self.dcnetv5 = conv(12 * 81, 32, kernel_size=3, stride=1, padding=1, dilation=1)  #
    self.dcnetv6 = conv(4, 32, kernel_size=3, stride=1, padding=1, dilation=1)  #
    if exp_unc:
        self.dcnet = bfmodule(128, 2)
    else:
        self.dcnet = bfmodule(128, 1)

def compute_mcv(im,
                dispo,
                dchange2, dc_unc,
                flow, flow_raft_fw_full, flow_raft_bw_full,
                cuda_device_name, bs, disc_aux, training, pseudo_training, debug, use_opencv):
        if not training or disc_aux[-1] == 2:
            ## pre-processing
            Kinv, Kinv_n = get_intrinsics(disc_aux[3], noise=False)
            # get full res flow/expansion inputs
            H, W = im.size()[2:4]

            tau = (-dchange2[:, 0]).exp().detach()

            # # use different number of correspondences for bg, obj segmentation and pose
            # if self.training:
            #     fscale = 1. / 4
            #     fscalex = 1. / 8
            #     fscaled = 1. / 1
            # else:
            #     fscale = 128. / H
            #     fscalex = 32. / H
            #     fscaled = 448. / H

            fscale = 1. / 4
            fscalex = 1. / 8
            fscaled = 1. / 1

            hp0o = torch.cat([torch.arange(0, W, out=torch.cuda.FloatTensor(device=cuda_device_name)).view(1, -1).repeat(H, 1)[np.newaxis],  # 1,2,H,W
                              torch.arange(0, H, out=torch.cuda.FloatTensor(device=cuda_device_name)).view(-1, 1).repeat(1, W)[np.newaxis]], 0)[np.newaxis]
            hp0o = hp0o.repeat(bs, 1, 1, 1)
            hp1o = hp0o + flow  # b,2,H,W

            hp1o_trans = hp1o.detach().clone().permute(0, 2, 3, 1)
            hp1o_trans[:, :, :, 0] = (hp1o_trans[:, :, :, 0] / (W - 1)) * 2. - 1.
            hp1o_trans[:, :, :, 1] = (hp1o_trans[:, :, :, 1] / (H - 1)) * 2. - 1.

            bwfw_warp = F.grid_sample(flow_raft_bw_full, hp1o_trans)
            bwfw_consistency = torch.sqrt(torch.sum((bwfw_warp + flow_raft_fw_full) ** 2, dim=1))

            # bwfw_consistency = F.interpolate(bwfw_consistency[:, np.newaxis], [H, W], mode='bilinear', align_corners=True).detach()[:, 0]

            # to deal with input resizing (TODO: move it inside intrinsics)

            # if not self.training:
            #     hp0o[:, 0] *= disc_aux[3][10]
            #     hp0o[:, 1] *= disc_aux[3][11]
            #     hp1o[:, 0] *= disc_aux[3][10]
            #     hp1o[:, 1] *= disc_aux[3][11]
            # else:
            #     hp0o[:, 0] *= disc_aux[3][10].reshape(bs,1,1)
            #     hp0o[:, 1] *= disc_aux[3][11].reshape(bs,1,1)
            #     hp1o[:, 0] *= disc_aux[3][10].reshape(bs,1,1)
            #     hp1o[:, 1] *= disc_aux[3][11].reshape(bs,1,1)
            hp0o[:, 0] *= disc_aux[3][10].reshape(bs, 1, 1)
            hp0o[:, 1] *= disc_aux[3][11].reshape(bs, 1, 1)
            hp1o[:, 0] *= disc_aux[3][10].reshape(bs, 1, 1)
            hp1o[:, 1] *= disc_aux[3][11].reshape(bs, 1, 1)

            # sample correspondence for object segmentation (fscaled)
            hp0d = F.interpolate(hp0o, scale_factor=fscaled, mode='nearest')
            hp1d = F.interpolate(hp1o, scale_factor=fscaled, mode='nearest')
            _, _, hd, wd = hp0d.shape
            hp0d = hp0d.view(bs, 2, -1).permute(0, 2, 1)
            hp1d = hp1d.view(bs, 2, -1).permute(0, 2, 1)
            hp0d = torch.cat((hp0d, torch.ones(bs, hp0d.shape[1], 1).cuda(cuda_device_name)), -1)
            hp1d = torch.cat((hp1d, torch.ones(bs, hp0d.shape[1], 1).cuda(cuda_device_name)), -1)
            uncd = torch.cat((F.interpolate(bwfw_consistency[:, np.newaxis], scale_factor=fscaled, mode='nearest'),
                              F.interpolate(dc_unc[:, np.newaxis].detach(), scale_factor=fscaled, mode='nearest')), 1)
            taud = F.interpolate(tau[:, np.newaxis], scale_factor=fscaled, mode='nearest').view(bs, 1, -1)

            # sample correspondence for fg/bg seg (fscale)
            # hp0 = F.interpolate(hp0o, scale_factor=fscale, mode='nearest')
            # hp1 = F.interpolate(hp1o, scale_factor=fscale, mode='nearest')
            # _, _, h, w = hp0.shape
            # hp0 = hp0.view(bs, 2, -1).permute(0, 2, 1)
            # hp1 = hp1.view(bs, 2, -1).permute(0, 2, 1)
            # hp0 = torch.cat((hp0, torch.ones(bs, hp0.shape[1], 1).cuda(cuda_device_name)), -1)
            # hp1 = torch.cat((hp1, torch.ones(bs, hp0.shape[1], 1).cuda(cuda_device_name)), -1)
            # unc = torch.cat((F.interpolate(bwfw_consistency[:, np.newaxis], scale_factor=fscale, mode='nearest'),
            #                  F.interpolate(dc_unc[:, np.newaxis].detach(), scale_factor=fscale, mode='nearest')), 1)
            # tau = F.interpolate(tau[:, np.newaxis], scale_factor=fscale, mode='nearest').view(bs, 1, -1)

            # sample correspondence for pose estimation (fscalex)
            hp0x = F.interpolate(hp0o, scale_factor=fscalex, mode='nearest')
            hp1x = F.interpolate(hp1o, scale_factor=fscalex, mode='nearest')
            hp0x = hp0x.view(bs, 2, -1).permute(0, 2, 1)
            hp1x = hp1x.view(bs, 2, -1).permute(0, 2, 1)
            hp0x = torch.cat((hp0x, torch.ones(bs, hp0x.shape[1], 1).cuda(cuda_device_name)), -1)
            hp1x = torch.cat((hp1x, torch.ones(bs, hp0x.shape[1], 1).cuda(cuda_device_name)), -1)

            ## camera pose estimation
            if pseudo_training:
                # use ground-truth rotation/translation with noise
                rot = disc_aux[-2][:, :3].detach().cuda(cuda_device_name)
                trans = disc_aux[-2][:, 3:].detach().cuda(cuda_device_name)
                if not debug:
                    rot = rot + torch.Tensor(np.random.normal(loc=0., scale=5e-4, size=(bs, 3))).cuda(cuda_device_name)
                    trans = trans + torch.Tensor(np.random.normal(loc=0., scale=5e-2, size=(bs, 3))).cuda(cuda_device_name) * trans.norm(2, 1)[:, np.newaxis]
                trans = trans / trans.norm(2, 1)[:, np.newaxis]
                Ex = get_skew_mat(trans.cpu(), rot.cpu())
            else:
                # estimate R/T with 5-pt algirhtm
                with torch.no_grad():
                    rand = False
                    unc_occ = F.interpolate(bwfw_consistency[:, np.newaxis], scale_factor=fscalex, mode='nearest').view(bs, -1)
                    rotx, transx, Ex = F_ngransac(hp0x, hp1x, Kinv.inverse(), rand, unc_occ, Kn=Kinv_n.inverse(), cv=use_opencv)
                    rot = rotx.cuda(cuda_device_name).detach()
                    trans = transx.cuda(cuda_device_name).detach()

            # self.computed_networks_outputs['rot'] = rot
            # self.computed_networks_outputs['Ex'] = Ex
            # self.computed_networks_outputs['trans'] = trans

            # ## fg/bg segmentation
            # # rigidity cost maps
            # mcost00, mcost01, mcost1, mcost2, mcost3, mcost4, p3dmag, _ = compute_geo_costs(rot, trans, Ex, Kinv, hp0, hp1, tau, Kinv_n=Kinv_n)
            # # depth contrast cost

            # self.computed_networks_outputs['dispo'] = dispo

            # ## rigid instance segmentation
            # cost compute
            mcost00, mcost01, mcost1, mcost2, mcost3, mcost4, p3dmag, _ = compute_geo_costs(rot, trans, Ex, Kinv, hp0d, hp1d, taud, Kinv_n=Kinv_n)
            disp = F.interpolate(dispo, [hd, wd], mode='bilinear', align_corners=True)
            med_dgt = torch.median(disp.view(bs, -1), dim=-1)[0]
            med_dp3d = torch.median(p3dmag.view(bs, -1), dim=-1)[0]
            med_ratio = (med_dgt / med_dp3d)[:, np.newaxis, np.newaxis, np.newaxis]
            log_dratio = (med_ratio * p3dmag.view(bs, 1, hd, wd) / disp.view(bs, 1, hd, wd)).log()
            # pseudo 3D point compute
            depth = (1. / disp).view(bs, 1, -1)
            depth = depth.clamp(depth.median() / 10, depth.median() * 10)
            p03d = depth * Kinv.matmul(hp0d.permute(0, 2, 1))
            p13d = depth / taud * Kinv_n.matmul(hp1d.permute(0, 2, 1))
            #            p13d = kornia.angle_axis_to_rotation_matrix(rot).matmul(p13d)  # remove rotation
            p13d = kornia.geometry.conversions.angle_axis_to_rotation_matrix(rot).matmul(p13d)  # remove rotation
            pts = torch.cat([p03d, p13d], -1)  # bs, 3, 2*N
            # normalize it
            for i in range(bs):
                pts[i] = pts[i] - pts[i].mean(-1, keepdims=True)  # zero mean
                pts[i] = pts[i] / pts[i].flatten().std()  # unit std
            p03d = pts[:, :, :p03d.shape[-1]]
            p13d = pts[:, :, p03d.shape[-1]:]

            costs = torch.cat((
                0.01 * (mcost00 + mcost01).view(bs, 1, hd, wd).detach(),
                2e3 * mcost1.view(bs, 1, hd, wd).detach(),
                mcost2.view(bs, 1, hd, wd).detach(),
                30 * mcost3.view(bs, 1, hd, wd).detach(),
                mcost4.view(bs, 1, hd, wd).detach(),
                0.2 * uncd[:, :1].view(bs, 1, hd, wd).detach(),
                0.2 * uncd[:, 1:].view(bs, 1, hd, wd).detach(),
                3 * log_dratio.view(bs, 1, hd, wd).detach(),
                p03d.view(bs, 3, hd, wd).detach(),
                p13d.view(bs, 3, hd, wd).detach(),
            ), 1)

            costs[torch.isnan(costs)] = 0.
            costs[torch.isneginf(costs)] = -100
            costs[torch.isposinf(costs)] = 1100
            return torch.clip(costs, min=-100, max=1100)

            # if (torch.isnan(costs.min())):
            #     print(hp0d)
            #     print(hp1d)
            # print(rot, trans, Ex)