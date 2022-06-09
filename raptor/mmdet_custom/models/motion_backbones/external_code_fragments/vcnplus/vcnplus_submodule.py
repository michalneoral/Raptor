from __future__ import print_function

import torch.utils.data
from kornia.geometry.conversions import angle_axis_to_rotation_matrix as aa2rm
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import tqdm

class WarpModule(nn.Module):
    """
    taken from https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
    """
    def __init__(self, size):
        super(WarpModule, self).__init__()
        B,W,H = size
        self.BWH =  [B,W,H]
        # mesh grid

        self.create_grid(B,W,H)


    def create_grid(self, B,W,H,cuda_device_name=None):
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

        if cuda_device_name is not None:
            self.register_buffer('grid', torch.cat((xx, yy), 1).float().cuda(cuda_device_name).detach())
        else:
            self.register_buffer('grid', torch.cat((xx, yy), 1).float().detach())

    def forward(self, x, flo, simple_test=True):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()

        cuda_device_name = x.device
        if self.BWH != [B, W, H]:
            self.BWH = [B, W, H]
            self.create_grid(B, W, H, cuda_device_name=cuda_device_name)

        if simple_test:
            vgrid = self.grid[0:B] + flo
        else:
            vgrid = self.grid + flo

        # scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)
        #output = nn.functional.grid_sample(x, vgrid)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = ((vgrid[:,:,:,0].abs()<1) * (vgrid[:,:,:,1].abs()<1)) >0
        return output*mask.unsqueeze(1).float(), mask


def get_grid(B, H, W, cuda_device_name=None):
    meshgrid_base = np.meshgrid(range(0, W), range(0, H))[::-1]
    basey = np.reshape(meshgrid_base[0], [1, 1, 1, H, W])
    basex = np.reshape(meshgrid_base[1], [1, 1, 1, H, W])
    if cuda_device_name is None:
        grid = torch.tensor(np.concatenate((basex.reshape((-1, H, W, 1)), basey.reshape((-1, H, W, 1))), -1)).cuda().float()
    else:
        grid = torch.tensor(np.concatenate((basex.reshape((-1, H, W, 1)), basey.reshape((-1, H, W, 1))), -1)).cuda(cuda_device_name).float()
    return grid.view(1, 1, H, W, 2)


class flow_reg(nn.Module):
    """
    Soft winner-take-all that selects the most likely diplacement.
    Set ent=True to enable entropy output.
    Set maxdisp to adjust maximum allowed displacement towards one side.
        maxdisp=4 searches for a 9x9 region.
    Set fac to squeeze search window.
        maxdisp=4 and fac=2 gives search window of 9x5
    """
    def __init__(self, size, ent=False, maxdisp = int(4), fac=1):
        B,W,H = size
        super(flow_reg, self).__init__()
        self.BWH = [B,W,H]
        self.ent = ent
        self.md = maxdisp
        self.fac = fac
        self.truncated = True
        self.wsize = 3  # by default using truncation 7x7

        self.create_buffers(B,W,H)

        self.pool3d = nn.MaxPool3d((self.wsize*2+1,self.wsize*2+1,1),stride=1,padding=(self.wsize,self.wsize,0))

    def create_buffers(self, B, W, H, cuda_device_name=None):
        flowrangey = range(-self.md, self.md + 1)
        flowrangex = range(-int(self.md // self.fac), int(self.md // self.fac) + 1)
        meshgrid = np.meshgrid(flowrangex, flowrangey)
        flowy = np.tile(np.reshape(meshgrid[0], [1, 2 * self.md + 1, 2 * int(self.md // self.fac) + 1, 1, 1]), (B, 1, 1, H, W))
        flowx = np.tile(np.reshape(meshgrid[1], [1, 2 * self.md + 1, 2 * int(self.md // self.fac) + 1, 1, 1]), (B, 1, 1, H, W))
        if cuda_device_name is not None:
            self.register_buffer('flowx', torch.cuda.FloatTensor(flowx, device=cuda_device_name).detach())
            self.register_buffer('flowy', torch.cuda.FloatTensor(flowy, device=cuda_device_name).detach())
        else:
            self.register_buffer('flowx', torch.Tensor(flowx).detach())
            self.register_buffer('flowy', torch.Tensor(flowy).detach())

    def forward(self, x, simple_test=False):
        cuda_device_name = x.device
        b,u,v,h,w = x.shape
        oldx = x
        if self.BWH != [b,w,h]:
            self.BWH = [b,w,h]
            self.create_buffers(b,w,h,cuda_device_name=cuda_device_name)

        if self.truncated:
            # truncated softmax
            x = x.view(b,u*v,h,w)

            idx = x.argmax(1)[:,np.newaxis]
            if x.is_cuda:
                mask = Variable(torch.cuda.HalfTensor(b,u*v,h,w,device=cuda_device_name)).fill_(0)
            else:
                mask = Variable(torch.FloatTensor(b,u*v,h,w)).fill_(0)
            mask.scatter_(1,idx,1)
            mask = mask.view(b,1,u,v,-1)
            mask = self.pool3d(mask)[:,0].view(b,u,v,h,w)

            ninf = x.clone().fill_(-np.inf).view(b,u,v,h,w)
            x = torch.where(mask.byte(),oldx,ninf)
        else:
            self.wsize = (np.sqrt(u*v)-1)/2

        b,u,v,h,w = x.shape
        x = F.softmax(x.view(b,-1,h,w),1).view(b,u,v,h,w)
        if np.isnan(x.min().detach().cpu()):
            #pdb.set_trace()
            x[torch.isnan(x)] = F.softmax(oldx[torch.isnan(x)])

        if simple_test:
            flowx_st = self.flowx[0:b]
            flowy_st = self.flowy[0:b]
            outx = torch.sum(torch.sum(x * flowx_st, 1), 1, keepdim=True)
            outy = torch.sum(torch.sum(x * flowy_st, 1), 1, keepdim=True)

        else:
            outx = torch.sum(torch.sum(x * self.flowx, 1), 1, keepdim=True)
            outy = torch.sum(torch.sum(x * self.flowy, 1), 1, keepdim=True)

        if self.ent:
            # local
            local_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
            if self.wsize == 0:
                local_entropy[:] = 1.
            else:
                local_entropy /= np.log((self.wsize*2+1)**2)

            # global
            x = F.softmax(oldx.view(b,-1,h,w),1).view(b,u,v,h,w)
            global_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
            global_entropy /= np.log(x.shape[1]*x.shape[2])
            return torch.cat([outx,outy],1),torch.cat([local_entropy, global_entropy],1)
        else:
            return torch.cat([outx,outy],1),None


def conv4d(data, filters, bias=None, permute_filters=True, use_half=False):
    """
    This is done by stacking results of multiple 3D convolutions, and is very slow.
    Taken from https://github.com/ignacio-rocco/ncnet
    """
    b, c, h, w, d, t = data.size()

    data = data.permute(2, 0, 1, 3, 4, 5).contiguous()  # permute to avoid making contiguous inside loop

    # Same permutation is done with filters, unless already provided with permutation
    if permute_filters:
        filters = filters.permute(2, 0, 1, 3, 4, 5).contiguous()  # permute to avoid making contiguous inside loop

    c_out = filters.size(1)
    if use_half:
        output = Variable(torch.HalfTensor(h, b, c_out, w, d, t), requires_grad=data.requires_grad)
    else:
        output = Variable(torch.zeros(h, b, c_out, w, d, t), requires_grad=data.requires_grad)

    padding = filters.size(0) // 2
    if use_half:
        Z = Variable(torch.zeros(padding, b, c, w, d, t).half())
    else:
        Z = Variable(torch.zeros(padding, b, c, w, d, t))

    if data.is_cuda:
        Z = Z.cuda(data.get_device())
        output = output.cuda(data.get_device())

    data_padded = torch.cat((Z, data, Z), 0)

    for i in range(output.size(0)):  # loop on first feature dimension
        # convolve with center channel of filter (at position=padding)
        output[i, :, :, :, :, :] = F.conv3d(data_padded[i + padding, :, :, :, :, :],
                                            filters[padding, :, :, :, :, :], bias=bias, stride=1, padding=padding)
        # convolve with upper/lower channels of filter (at postions [:padding] [padding+1:])
        for p in range(1, padding + 1):
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(data_padded[i + padding - p, :, :, :, :, :],
                                                                           filters[padding - p, :, :, :, :, :], bias=None, stride=1, padding=padding)
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(data_padded[i + padding + p, :, :, :, :, :],
                                                                           filters[padding + p, :, :, :, :, :], bias=None, stride=1, padding=padding)

    output = output.permute(1, 2, 0, 3, 4, 5).contiguous()
    return output


class Conv4d(_ConvNd):
    """Applies a 4D convolution over an input signal composed of several input
    planes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, pre_permuted_filters=True):
        # stride, dilation and groups !=1 functionality not tested
        stride = 1
        dilation = 1
        groups = 1
        # zero padding is added automatically in conv4d function to preserve tensor size
        padding = 0
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)
        super(Conv4d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _quadruple(0), groups, bias)
        # weights will be sliced along one dimension during convolution loop
        # make the looping dimension to be the first one in the tensor,
        # so that we don't need to call contiguous() inside the loop
        self.pre_permuted_filters = pre_permuted_filters
        if self.pre_permuted_filters:
            self.weight.data = self.weight.data.permute(2, 0, 1, 3, 4, 5).contiguous()
        self.use_half = False

    #    self.isbias = bias
    #    if not self.isbias:
    #        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, input):
        out = conv4d(input, self.weight, bias=self.bias, permute_filters=not self.pre_permuted_filters, use_half=self.use_half)  # filters pre-permuted in constructor
        #    if not self.isbias:
        #        b,c,u,v,h,w = out.shape
        #        out = self.bn(out.view(b,c,-1)).view(b,c,u,v,h,w)
        return out


class fullConv4d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, pre_permuted_filters=True):
        super(fullConv4d, self).__init__()
        self.conv = Conv4d(in_channels, out_channels, kernel_size, bias=bias, pre_permuted_filters=pre_permuted_filters)
        self.isbias = bias
        if not self.isbias:
            self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, input):
        out = self.conv(input)
        if not self.isbias:
            b, c, u, v, h, w = out.shape
            out = self.bn(out.view(b, c, -1)).view(b, c, u, v, h, w)
        return out


class butterfly4D(torch.nn.Module):
    '''
    butterfly 4d
    '''

    def __init__(self, fdima, fdimb, withbn=True, full=True, groups=1):
        super(butterfly4D, self).__init__()
        self.proj = nn.Sequential(projfeat4d(fdima, fdimb, 1, with_bn=withbn, groups=groups),
                                  nn.ReLU(inplace=True), )
        self.conva1 = sepConv4dBlock(fdimb, fdimb, with_bn=withbn, stride=(2, 1, 1), full=full, groups=groups)
        self.conva2 = sepConv4dBlock(fdimb, fdimb, with_bn=withbn, stride=(2, 1, 1), full=full, groups=groups)
        self.convb3 = sepConv4dBlock(fdimb, fdimb, with_bn=withbn, stride=(1, 1, 1), full=full, groups=groups)
        self.convb2 = sepConv4dBlock(fdimb, fdimb, with_bn=withbn, stride=(1, 1, 1), full=full, groups=groups)
        self.convb1 = sepConv4dBlock(fdimb, fdimb, with_bn=withbn, stride=(1, 1, 1), full=full, groups=groups)

    # @profile
    def forward(self, x):
        out = self.proj(x)
        b, c, u, v, h, w = out.shape  # 9x9

        out1 = self.conva1(out)  # 5x5, 3
        _, c1, u1, v1, h1, w1 = out1.shape

        out2 = self.conva2(out1)  # 3x3, 9
        _, c2, u2, v2, h2, w2 = out2.shape

        out2 = self.convb3(out2)  # 3x3, 9

        tout1 = F.interpolate(out2.view(b, c, u2, v2, -1), (u1, v1, h2 * w2), mode='trilinear', align_corners=True).view(b, c, u1, v1, h2, w2)  # 5x5
        tout1 = F.interpolate(tout1.view(b, c, -1, h2, w2), (u1 * v1, h1, w1), mode='trilinear', align_corners=True).view(b, c, u1, v1, h1, w1)  # 5x5
        out1 = tout1 + out1
        out1 = self.convb2(out1)

        tout = F.interpolate(out1.view(b, c, u1, v1, -1), (u, v, h1 * w1), mode='trilinear', align_corners=True).view(b, c, u, v, h1, w1)
        tout = F.interpolate(tout.view(b, c, -1, h1, w1), (u * v, h, w), mode='trilinear', align_corners=True).view(b, c, u, v, h, w)
        out = tout + out
        out = self.convb1(out)

        return out


class projfeat4d(torch.nn.Module):
    '''
    Turn 3d projection into 2d projection
    '''

    def __init__(self, in_planes, out_planes, stride, with_bn=True, groups=1):
        super(projfeat4d, self).__init__()
        self.with_bn = with_bn
        self.stride = stride
        self.conv1 = nn.Conv3d(in_planes, out_planes, 1, (stride, stride, 1), padding=0, bias=not with_bn, groups=groups)
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        b, c, u, v, h, w = x.size()
        x = self.conv1(x.view(b, c, u, v, h * w))
        if self.with_bn:
            x = self.bn(x)
        _, c, u, v, _ = x.shape
        x = x.view(b, c, u, v, h, w)
        return x


class sepConv4d(torch.nn.Module):
    '''
    Separable 4d convolution block as 2 3D convolutions
    '''

    def __init__(self, in_planes, out_planes, stride=(1, 1, 1), with_bn=True, ksize=3, full=True, groups=1):
        super(sepConv4d, self).__init__()
        bias = not with_bn
        self.isproj = False
        self.stride = stride[0]
        expand = 1

        if with_bn:
            if in_planes != out_planes:
                self.isproj = True
                self.proj = nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, bias=bias, padding=0, groups=groups),
                                          nn.BatchNorm2d(out_planes))
            if full:
                self.conv1 = nn.Sequential(
                    nn.Conv3d(in_planes * expand, in_planes, (1, ksize, ksize), stride=(1, self.stride, self.stride), bias=bias, padding=(0, ksize // 2, ksize // 2),
                              groups=groups),
                    nn.BatchNorm3d(in_planes))
            else:
                self.conv1 = nn.Sequential(nn.Conv3d(in_planes * expand, in_planes, (1, ksize, ksize), stride=1, bias=bias, padding=(0, ksize // 2, ksize // 2), groups=groups),
                                           nn.BatchNorm3d(in_planes))
            self.conv2 = nn.Sequential(
                nn.Conv3d(in_planes, in_planes * expand, (ksize, ksize, 1), stride=(self.stride, self.stride, 1), bias=bias, padding=(ksize // 2, ksize // 2, 0), groups=groups),
                nn.BatchNorm3d(in_planes * expand))
        else:
            if in_planes != out_planes:
                self.isproj = True
                self.proj = nn.Conv2d(in_planes, out_planes, 1, bias=bias, padding=0, groups=groups)
            if full:
                self.conv1 = nn.Conv3d(in_planes * expand, in_planes, (1, ksize, ksize), stride=(1, self.stride, self.stride), bias=bias, padding=(0, ksize // 2, ksize // 2),
                                       groups=groups)
            else:
                self.conv1 = nn.Conv3d(in_planes * expand, in_planes, (1, ksize, ksize), stride=1, bias=bias, padding=(0, ksize // 2, ksize // 2), groups=groups)
            self.conv2 = nn.Conv3d(in_planes, in_planes * expand, (ksize, ksize, 1), stride=(self.stride, self.stride, 1), bias=bias, padding=(ksize // 2, ksize // 2, 0),
                                   groups=groups)
        self.relu = nn.ReLU(inplace=True)

    # @profile
    def forward(self, x):
        b, c, u, v, h, w = x.shape
        x = self.conv2(x.view(b, c, u, v, -1))
        b, c, u, v, _ = x.shape
        x = self.relu(x)
        x = self.conv1(x.view(b, c, -1, h, w))
        b, c, _, h, w = x.shape

        if self.isproj:
            x = self.proj(x.view(b, c, -1, w))
        x = x.view(b, -1, u, v, h, w)
        return x


class sepConv4dBlock(torch.nn.Module):
    '''
    Separable 4d convolution block as 2 2D convolutions and a projection
    layer
    '''

    def __init__(self, in_planes, out_planes, stride=(1, 1, 1), with_bn=True, full=True, groups=1):
        super(sepConv4dBlock, self).__init__()
        if in_planes == out_planes and stride == (1, 1, 1):
            self.downsample = None
        else:
            if full:
                self.downsample = sepConv4d(in_planes, out_planes, stride, with_bn=with_bn, ksize=1, full=full, groups=groups)
            else:
                self.downsample = projfeat4d(in_planes, out_planes, stride[0], with_bn=with_bn, groups=groups)
        self.conv1 = sepConv4d(in_planes, out_planes, stride, with_bn=with_bn, full=full, groups=groups)
        self.conv2 = sepConv4d(out_planes, out_planes, (1, 1, 1), with_bn=with_bn, full=full, groups=groups)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    # @profile
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        if self.downsample:
            x = self.downsample(x)
        out = self.relu2(x + self.conv2(out))
        return out


class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None, dilation=1, with_bn=True):
        super(residualBlock, self).__init__()
        if dilation > 1:
            padding = dilation
        else:
            padding = 1

        if with_bn:
            self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3, stride, padding, dilation=dilation)
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1)
        else:
            self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3, stride, padding, dilation=dilation, with_bn=False)
            self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, with_bn=False)
        self.downsample = downsample
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, inplace=True))


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()
        bias = not with_bn

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters)), )
        else:
            self.cb_unit = nn.Sequential(conv_mod, )

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()
        bias = not with_bn
        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.LeakyReLU(0.1, inplace=True), )
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.LeakyReLU(0.1, inplace=True), )

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class pyramidPooling(nn.Module):

    def __init__(self, in_channels, with_bn=True, levels=4):
        super(pyramidPooling, self).__init__()
        self.levels = levels

        self.paths = []
        for i in range(levels):
            self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, with_bn=with_bn))
        self.path_module_list = nn.ModuleList(self.paths)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        for pool_size in np.linspace(1, min(h, w) // 2, self.levels, dtype=int):
            k_sizes.append((int(h / pool_size), int(w / pool_size)))
            strides.append((int(h / pool_size), int(w / pool_size)))
        k_sizes = k_sizes[::-1]
        strides = strides[::-1]

        pp_sum = x

        for i, module in enumerate(self.path_module_list):
            out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
            out = module(out)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            pp_sum = pp_sum + 1. / self.levels * out
        pp_sum = self.relu(pp_sum / 2.)

        return pp_sum


class pspnet(nn.Module):
    """
    Modified PSPNet.  https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/pspnet.py
    """

    def __init__(self, is_proj=True, groups=1):
        super(pspnet, self).__init__()
        self.inplanes = 32
        self.is_proj = is_proj

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=16,
                                                 padding=1, stride=2)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=16,
                                                 padding=1, stride=1)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock, 64, 1, stride=2)
        self.res_block5 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.res_block6 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.res_block7 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)

        # Iconvs
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=True),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                         padding=1, stride=1))
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                          padding=1, stride=1)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=True),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                         padding=1, stride=1))
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                          padding=1, stride=1)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=True),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                         padding=1, stride=1))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                          padding=1, stride=1)
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=True),
                                     conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                                         padding=1, stride=1))
        self.iconv2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64,
                                          padding=1, stride=1)

        if self.is_proj:
            self.proj6 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=128 // groups, padding=0, stride=1)
            self.proj5 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=128 // groups, padding=0, stride=1)
            self.proj4 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=128 // groups, padding=0, stride=1)
            self.proj3 = conv2DBatchNormRelu(in_channels=64, k_size=1, n_filters=64 // groups, padding=0, stride=1)
            self.proj2 = conv2DBatchNormRelu(in_channels=64, k_size=1, n_filters=64 // groups, padding=0, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # H, W -> H/2, W/2
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)

        ## H/2, W/2 -> H/4, W/4
        pool1 = F.max_pool2d(conv1, 3, 2, 1)

        # H/4, W/4 -> H/16, W/16
        rconv3 = self.res_block3(pool1)
        conv4 = self.res_block5(rconv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)

        conv6x = F.interpolate(conv6, [conv5.size()[2], conv5.size()[3]], mode='bilinear', align_corners=True)
        concat5 = torch.cat((conv5, self.upconv6[1](conv6x)), dim=1)
        conv5 = self.iconv5(concat5)

        conv5x = F.interpolate(conv5, [conv4.size()[2], conv4.size()[3]], mode='bilinear', align_corners=True)
        concat4 = torch.cat((conv4, self.upconv5[1](conv5x)), dim=1)
        conv4 = self.iconv4(concat4)

        conv4x = F.interpolate(conv4, [rconv3.size()[2], rconv3.size()[3]], mode='bilinear', align_corners=True)
        concat3 = torch.cat((rconv3, self.upconv4[1](conv4x)), dim=1)
        conv3 = self.iconv3(concat3)

        conv3x = F.interpolate(conv3, [pool1.size()[2], pool1.size()[3]], mode='bilinear', align_corners=True)
        concat2 = torch.cat((pool1, self.upconv3[1](conv3x)), dim=1)
        conv2 = self.iconv2(concat2)

        if self.is_proj:
            proj6 = self.proj6(conv6)
            proj5 = self.proj5(conv5)
            proj4 = self.proj4(conv4)
            proj3 = self.proj3(conv3)
            proj2 = self.proj2(conv2)
            return proj6, proj5, proj4, proj3, proj2
        else:
            return conv6, conv5, conv4, conv3, conv2


class pspnet_s(nn.Module):
    """
    Modified PSPNet.  https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/pspnet.py
    """

    def __init__(self, is_proj=True, groups=1):
        super(pspnet_s, self).__init__()
        self.inplanes = 32
        self.is_proj = is_proj

        # Encoder
        self.convbnrelu1_1 = conv2DBatchNormRelu(in_channels=3, k_size=3, n_filters=16,
                                                 padding=1, stride=2)
        self.convbnrelu1_2 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=16,
                                                 padding=1, stride=1)
        self.convbnrelu1_3 = conv2DBatchNormRelu(in_channels=16, k_size=3, n_filters=32,
                                                 padding=1, stride=1)
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock, 64, 1, stride=2)
        self.res_block5 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.res_block6 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.res_block7 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)

        # Iconvs
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=True),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                         padding=1, stride=1))
        self.iconv5 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                          padding=1, stride=1)
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=True),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                         padding=1, stride=1))
        self.iconv4 = conv2DBatchNormRelu(in_channels=192, k_size=3, n_filters=128,
                                          padding=1, stride=1)
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=True),
                                     conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                                         padding=1, stride=1))
        self.iconv3 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                          padding=1, stride=1)
        # self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2, align_corners=True),
        #                             conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
        #                                         padding=1, stride=1))
        # self.iconv2 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=64,
        #                                         padding=1, stride=1)

        if self.is_proj:
            self.proj6 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=128 // groups, padding=0, stride=1)
            self.proj5 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=128 // groups, padding=0, stride=1)
            self.proj4 = conv2DBatchNormRelu(in_channels=128, k_size=1, n_filters=128 // groups, padding=0, stride=1)
            self.proj3 = conv2DBatchNormRelu(in_channels=64, k_size=1, n_filters=64 // groups, padding=0, stride=1)
            # self.proj2 = conv2DBatchNormRelu(in_channels=64, k_size=1,n_filters=64//groups, padding=0,stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # H, W -> H/2, W/2
        conv1 = self.convbnrelu1_1(x)
        conv1 = self.convbnrelu1_2(conv1)
        conv1 = self.convbnrelu1_3(conv1)

        ## H/2, W/2 -> H/4, W/4
        pool1 = F.max_pool2d(conv1, 3, 2, 1)

        # H/4, W/4 -> H/16, W/16
        rconv3 = self.res_block3(pool1)
        conv4 = self.res_block5(rconv3)
        conv5 = self.res_block6(conv4)
        conv6 = self.res_block7(conv5)
        conv6 = self.pyramid_pooling(conv6)

        conv6x = F.interpolate(conv6, [conv5.size()[2], conv5.size()[3]], mode='bilinear', align_corners=True)
        concat5 = torch.cat((conv5, self.upconv6[1](conv6x)), dim=1)
        conv5 = self.iconv5(concat5)

        conv5x = F.interpolate(conv5, [conv4.size()[2], conv4.size()[3]], mode='bilinear', align_corners=True)
        concat4 = torch.cat((conv4, self.upconv5[1](conv5x)), dim=1)
        conv4 = self.iconv4(concat4)

        conv4x = F.interpolate(conv4, [rconv3.size()[2], rconv3.size()[3]], mode='bilinear', align_corners=True)
        concat3 = torch.cat((rconv3, self.upconv4[1](conv4x)), dim=1)
        conv3 = self.iconv3(concat3)

        # conv3x = F.interpolate(conv3, [pool1.size()[2],pool1.size()[3]],mode='bilinear', align_corners=True)
        # concat2 = torch.cat((pool1,self.upconv3[1](conv3x)),dim=1)
        # conv2 = self.iconv2(concat2)

        if self.is_proj:
            proj6 = self.proj6(conv6)
            proj5 = self.proj5(conv5)
            proj4 = self.proj4(conv4)
            proj3 = self.proj3(conv3)
            #    proj2 = self.proj2(conv2)
            #    return proj6,proj5,proj4,proj3,proj2
            return proj6, proj5, proj4, proj3
        else:
            #    return conv6, conv5, conv4, conv3, conv2
            return conv6, conv5, conv4, conv3


class bfmodule(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(bfmodule, self).__init__()
        self.proj = conv2DBatchNormRelu(in_channels=inplanes, k_size=1, n_filters=64, padding=0, stride=1)
        self.inplanes = 64
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock, 64, 1, stride=2)
        self.res_block5 = self._make_layer(residualBlock, 64, 1, stride=2)
        self.res_block6 = self._make_layer(residualBlock, 64, 1, stride=2)
        self.res_block7 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)
        # Iconvs
        self.upconv6 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                           padding=1, stride=1)
        self.upconv5 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                           padding=1, stride=1)
        self.upconv4 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                           padding=1, stride=1)
        self.upconv3 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                           padding=1, stride=1)
        self.iconv5 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                          padding=1, stride=1)
        self.iconv4 = conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
                                          padding=1, stride=1)
        self.iconv3 = conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
                                          padding=1, stride=1)
        self.iconv2 = nn.Sequential(conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
                                                        padding=1, stride=1),
                                    nn.Conv2d(64, outplanes, kernel_size=3, stride=1, padding=1, bias=True))

        self.proj6 = nn.Conv2d(128, outplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.proj5 = nn.Conv2d(64, outplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.proj4 = nn.Conv2d(64, outplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.proj3 = nn.Conv2d(64, outplanes, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        proj = self.proj(x)  # 4x
        rconv3 = self.res_block3(proj)  # 8x
        conv4 = self.res_block5(rconv3)  # 16x
        conv5 = self.res_block6(conv4)  # 32x
        conv6 = self.res_block7(conv5)  # 64x
        conv6 = self.pyramid_pooling(conv6)  # 64x
        pred6 = self.proj6(conv6)

        conv6u = F.interpolate(conv6, [conv5.size()[2], conv5.size()[3]], mode='bilinear', align_corners=True)
        concat5 = torch.cat((conv5, self.upconv6(conv6u)), dim=1)
        conv5 = self.iconv5(concat5)  # 32x
        pred5 = self.proj5(conv5)

        conv5u = F.interpolate(conv5, [conv4.size()[2], conv4.size()[3]], mode='bilinear', align_corners=True)
        concat4 = torch.cat((conv4, self.upconv5(conv5u)), dim=1)
        conv4 = self.iconv4(concat4)  # 16x
        pred4 = self.proj4(conv4)

        conv4u = F.interpolate(conv4, [rconv3.size()[2], rconv3.size()[3]], mode='bilinear', align_corners=True)
        concat3 = torch.cat((rconv3, self.upconv4(conv4u)), dim=1)
        conv3 = self.iconv3(concat3)  # 8x
        pred3 = self.proj3(conv3)

        conv3u = F.interpolate(conv3, [x.size()[2], x.size()[3]], mode='bilinear', align_corners=True)
        concat2 = torch.cat((proj, self.upconv3(conv3u)), dim=1)
        pred2 = self.iconv2(concat2)  # 4x

        return pred2, pred3, pred4, pred5, pred6


class bfmodule_feat(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(bfmodule_feat, self).__init__()
        self.proj = conv2DBatchNormRelu(in_channels=inplanes, k_size=1, n_filters=64, padding=0, stride=1)
        self.inplanes = 64
        # Vanilla Residual Blocks
        self.res_block3 = self._make_layer(residualBlock, 64, 1, stride=2)
        self.res_block5 = self._make_layer(residualBlock, 64, 1, stride=2)
        self.res_block6 = self._make_layer(residualBlock, 64, 1, stride=2)
        self.res_block7 = self._make_layer(residualBlock, 128, 1, stride=2)
        self.pyramid_pooling = pyramidPooling(128, levels=3)
        # Iconvs
        self.upconv6 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                           padding=1, stride=1)
        self.upconv5 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                           padding=1, stride=1)
        self.upconv4 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                           padding=1, stride=1)
        self.upconv3 = conv2DBatchNormRelu(in_channels=64, k_size=3, n_filters=32,
                                           padding=1, stride=1)
        self.iconv5 = conv2DBatchNormRelu(in_channels=128, k_size=3, n_filters=64,
                                          padding=1, stride=1)
        self.iconv4 = conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
                                          padding=1, stride=1)
        self.iconv3 = conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
                                          padding=1, stride=1)
        self.iconv2 = conv2DBatchNormRelu(in_channels=96, k_size=3, n_filters=64,
                                          padding=1, stride=1)

        self.proj6 = nn.Conv2d(128, outplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.proj5 = nn.Conv2d(64, outplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.proj4 = nn.Conv2d(64, outplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.proj3 = nn.Conv2d(64, outplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.proj2 = nn.Conv2d(64, outplanes, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        proj = self.proj(x)  # 4x
        rconv3 = self.res_block3(proj)  # 8x
        conv4 = self.res_block5(rconv3)  # 16x
        conv5 = self.res_block6(conv4)  # 32x
        conv6 = self.res_block7(conv5)  # 64x
        conv6 = self.pyramid_pooling(conv6)  # 64x
        pred6 = self.proj6(conv6)

        conv6u = F.interpolate(conv6, [conv5.size()[2], conv5.size()[3]], mode='bilinear', align_corners=True)
        concat5 = torch.cat((conv5, self.upconv6(conv6u)), dim=1)
        conv5 = self.iconv5(concat5)  # 32x
        pred5 = self.proj5(conv5)

        conv5u = F.interpolate(conv5, [conv4.size()[2], conv4.size()[3]], mode='bilinear', align_corners=True)
        concat4 = torch.cat((conv4, self.upconv5(conv5u)), dim=1)
        conv4 = self.iconv4(concat4)  # 16x
        pred4 = self.proj4(conv4)

        conv4u = F.interpolate(conv4, [rconv3.size()[2], rconv3.size()[3]], mode='bilinear', align_corners=True)
        concat3 = torch.cat((rconv3, self.upconv4(conv4u)), dim=1)
        conv3 = self.iconv3(concat3)  # 8x
        pred3 = self.proj3(conv3)

        conv3u = F.interpolate(conv3, [x.size()[2], x.size()[3]], mode='bilinear', align_corners=True)
        concat2 = torch.cat((proj, self.upconv3(conv3u)), dim=1)
        conv2 = self.iconv2(concat2)  # 4x
        pred2 = self.proj2(conv2)  # 4x
        return pred2, conv2


def compute_geo_costs(rot, trans, Ex, Kinv, hp0, hp1, tau, Kinv_n=None):
    if Kinv_n is None: Kinv_n = Kinv
    #R01 = kornia.angle_axis_to_rotation_matrix(rot)
    R01 = aa2rm(rot)
    H01 = Kinv.inverse().matmul(R01).matmul(Kinv_n)
    comp_hp1 = H01.matmul(hp1.permute(0, 2, 1))
    foe = (comp_hp1 - tau * hp0.permute(0, 2, 1))
    parallax3d = Kinv.matmul(foe)
    p3dmag = parallax3d.norm(2, 1)[:, np.newaxis]
    parallax2d = (comp_hp1 / comp_hp1[:, -1:] - hp0.permute(0, 2, 1))[:, :2]
    p2dmag = parallax2d.norm(2, 1)[:, np.newaxis]
    p2dnorm = parallax2d / (1e-9 + p2dmag)
    foe_cam = Kinv.inverse().matmul(trans[:, :, np.newaxis])
    foe_cam = foe_cam[:, :2] / (1e-9 + foe_cam[:, -1:])
    direct = foe_cam - hp0.permute(0, 2, 1)[:, :2]
    directn = direct / (1e-9 + direct.norm(2, 1)[:, np.newaxis])

    # cost metrics: 0) R-homography+symterr; 1) sampson 2) 2D angular (P+P) 3) 3D distance 4) 3D angular (P+P)
    ##TODO validate
    comp_hp0 = H01.inverse().matmul(hp0.permute(0, 2, 1))
    mcost00 = parallax2d.norm(2, 1)
    mcost01 = (comp_hp0 / comp_hp0[:, -1:] - hp1.permute(0, 2, 1))[:, :2].norm(2, 1)
    mcost1 = sampson_err(Kinv.matmul(hp0.permute(0, 2, 1)),
                         Kinv_n.matmul(hp1.permute(0, 2, 1)), Ex.cuda(hp0.device).permute(0, 2, 1))  # variable K
    mcost2 = -(trans[:, -1:, np.newaxis]).sign() * (directn * p2dnorm).sum(1, keepdims=True)
    mcost4 = -(trans[:, :, np.newaxis] * parallax3d).sum(1, keepdims=True) / (p3dmag + 1e-9)
    mcost3 = torch.clamp(1 - mcost4.pow(2), 0, 1).sqrt() * p3dmag * mcost4.sign()
    # mcost10 = torch.clamp(1 - mcost2.pow(2), 0, 1).sqrt() * p2dmag * mcost2.sign()
    mcost10 = None
    return mcost00, mcost01, mcost1, mcost2, mcost3, mcost4, p3dmag, mcost10


def get_skew_mat(transx, rotx):
    #rot = kornia.angle_axis_to_rotation_matrix(rotx)
    #rot = kornia.geometry.conversions.angle_axis_to_rotation_matrix(rotx)
    rot = aa2rm(rotx)

    # transx = -rot.permute(0, 2, 1).matmul(transx[:, :, np.newaxis])[:, :, 0]
    rot = rot.permute(0, 2, 1)
    tx = torch.zeros(transx.shape[0], 3, 3)
    tx[:, 0, 1] = -transx[:, 2]
    tx[:, 0, 2] = transx[:, 1]
    tx[:, 1, 0] = transx[:, 2]
    tx[:, 1, 2] = -transx[:, 0]
    tx[:, 2, 0] = -transx[:, 1]
    tx[:, 2, 1] = transx[:, 0]
    return rot.matmul(tx)


def sampson_err(x1h, x2h, F):
    l2 = F.permute(0, 2, 1).matmul(x1h)
    l1 = F.matmul(x2h)
    algdis = (l1 * x1h).sum(1)
    dis = algdis ** 2 / (1e-9 + l1[:, 0] ** 2 + l1[:, 1] ** 2 + l2[:, 0] ** 2 + l2[:, 1] ** 2)
    return dis


def get_intrinsics(intr, noise=False):
    f = intr[0].float()
    cx = intr[1].float()
    cy = intr[2].float()
    if len(intr) > 10:  # test time
        dfx = intr[10].float()
        dfy = intr[11].float()
        dfx = 1.
        dfy = 1.
    else:  # train time
        dfx = 1.
        dfy = 1.
    bs = f.shape[0]

    delta = 1e-4
    if noise:
        fo = f.clone()
        cxo = cx.clone()
        cyo = cy.clone()
        f = torch.Tensor(np.random.normal(loc=0., scale=delta, size=(bs,))).cuda(f.device).exp() * fo
        cx = torch.Tensor(np.random.normal(loc=0., scale=delta, size=(bs,))).cuda(f.device).exp() * cxo
        cy = torch.Tensor(np.random.normal(loc=0., scale=delta, size=(bs,))).cuda(f.device).exp() * cyo

    # Kinv = torch.Tensor(np.eye(3)[np.newaxis]).cuda().repeat(bs,1,1)
    # Kinv[:,2,2] *= f
    # Kinv[:,0,2] -= cx
    # Kinv[:,1,2] -= cy
    # Kinv /= f[:,np.newaxis,np.newaxis] #4,3,3

    Kinv = torch.Tensor(np.eye(3)[np.newaxis]).cuda(f.device).repeat(bs, 1, 1)
    Kinv[:, 0, 0] = f / dfx
    Kinv[:, 1, 1] = f / dfy
    Kinv[:, 0, 2] = cx / dfx
    Kinv[:, 1, 2] = cy / dfy
    Kinv = Kinv.inverse()

    Taug = torch.cat(intr[4:10], -1).view(-1, bs).T.cuda(f.device)  # 4,6
    Taug = torch.cat((Taug.view(bs, 3, 2).permute(0, 2, 1), Kinv[:, 2:3]), 1)

    #print(Taug)

    Kinv = Kinv.matmul(Taug)
    if len(intr) > 12:
        Kinv_n = torch.Tensor(np.eye(3)[np.newaxis]).cuda(f.device).repeat(bs, 1, 1)
        fn = intr[12].float()
        # Kinv_n[:,2,2] *= fn
        # Kinv_n[:,0,2] -= cx
        # Kinv_n[:,1,2] -= cy
        # Kinv_n /= fn[:,np.newaxis,np.newaxis] #4,3,3
        Kinv_n = torch.Tensor(np.eye(3)[np.newaxis]).cuda(f.device).repeat(bs, 1, 1)
        Kinv_n[:, 0, 0] = fn / dfx
        Kinv_n[:, 1, 1] = fn / dfy
        Kinv_n[:, 0, 2] = cx / dfx
        Kinv_n[:, 1, 2] = cy / dfy
        Kinv_n = Kinv_n.inverse()

    elif noise:
        f = torch.Tensor(np.random.normal(loc=0., scale=delta, size=(bs,))).cuda(f.device).exp() * fo
        cx = torch.Tensor(np.random.normal(loc=0., scale=delta, size=(bs,))).cuda(f.device).exp() * cxo
        cy = torch.Tensor(np.random.normal(loc=0., scale=delta, size=(bs,))).cuda(f.device).exp() * cyo

        Kinv_n = torch.Tensor(np.eye(3)[np.newaxis]).cuda(f.device).repeat(bs, 1, 1)
        Kinv_n[:, 2, 2] *= f
        Kinv_n[:, 0, 2] -= cx
        Kinv_n[:, 1, 2] -= cy
        Kinv_n /= f[:, np.newaxis, np.newaxis]  # 4,3,3

        Taug = torch.cat(intr[4:10], -1).view(-1, bs).T  # 4,6
        Taug = torch.cat((Taug.view(bs, 3, 2).permute(0, 2, 1), Kinv_n[:, 2:3]), 1)
        Kinv_n = Kinv_n.matmul(Taug)
    else:
        Kinv_n = Kinv

    return Kinv, Kinv_n


def F_ngransac(hp0, hp1, Ks, rand, unc_occ, iters=1000, cv=False, Kn=None):
    if cv:
        tqdm.tqdm.write('Using OpenCV for Essential camera matrix (R,t)')
    if Kn is None:
        Kn = Ks
    import cv2

    b = hp1.shape[0]
    hp0_cpu = np.asarray(hp0.cpu())
    hp1_cpu = np.asarray(hp1.cpu())
    if not rand:
        ## TODO
        fmask = np.ones(hp0.shape[1]).astype(bool)
        rand_seed = 0
    else:
        fmask = np.random.choice([True, False], size=hp0.shape[1], p=[0.1, 0.9])
        rand_seed = np.random.randint(0, 1000)  # random seed to by used in C++
    ### TODO
    hp0 = Ks.inverse().matmul(hp0.permute(0, 2, 1)).permute(0, 2, 1)
    hp1 = Kn.inverse().matmul(hp1.permute(0, 2, 1)).permute(0, 2, 1)
    ratios = torch.zeros(hp0[:1, :, :1].shape)
    probs = torch.Tensor(np.ones(fmask.sum())) / fmask.sum()
    probs = probs[np.newaxis, :, np.newaxis]

    # probs = torch.Tensor(np.zeros(fmask.sum()))
    ##unc_occ = unc_occ<0; probs[unc_occ[0]] = 1./unc_occ.float().sum()
    # probs = F.softmax(-0.1*unc_occ[0],-1).cpu()
    # probs = probs[np.newaxis,:,np.newaxis]

    Es = torch.zeros((b, 3, 3)).float()  # estimated model
    rot = torch.zeros((b, 3)).float()  # estimated model
    trans = torch.zeros((b, 3)).float()  # estimated model
    out_model = torch.zeros((3, 3)).float()  # estimated model
    out_inliers = torch.zeros(probs.size())  # inlier mask of estimated model
    out_gradients = torch.zeros(probs.size())  # gradient tensor (only used during training)

    for i in range(b):
        pts1 = hp0[i:i + 1, fmask, :2].cpu()
        pts2 = hp1[i:i + 1, fmask, :2].cpu()
        # create data tensor of feature coordinates and matching ratios
        correspondences = torch.cat((pts1, pts2, ratios), axis=2)
        correspondences = correspondences.permute(2, 1, 0)
        # incount = ngransac.find_fundamental_mat(correspondences, probs, rand_seed, 1000, 0.1, True, out_model, out_inliers, out_gradients)
        # E = K1.T.dot(out_model).dot(K0)

        if cv == True:
            E, ffmask = cv2.findEssentialMat(np.asarray(pts1[0]), np.asarray(pts2[0]), np.eye(3), cv2.FM_RANSAC, threshold=0.0001)
            ffmask = ffmask[:, 0]
            Es[i] = torch.Tensor(E)
        else:
            import ngransac
            incount = ngransac.find_essential_mat(correspondences, probs, rand_seed, iters, 0.0001, out_model, out_inliers, out_gradients)
            Es[i] = out_model
            E = np.asarray(out_model)
            maskk = np.asarray(out_inliers[0, :, 0])
            ffmask = fmask.copy()
            ffmask[fmask] = maskk
        K1 = np.asarray(Kn[i].cpu())
        K0 = np.asarray(Ks[i].cpu())
        R1, R2, T = cv2.decomposeEssentialMat(E)
        for rott in [(R1, T), (R2, T), (R1, -T), (R2, -T)]:
            if testEss(K0, K1, rott[0], rott[1], hp0_cpu[0, ffmask].T, hp1_cpu[i, ffmask].T):
                # if testEss(K0,K1,rott[0],rott[1],hp0_cpu[0,ffmask].T[:,ffmask.sum()//10::ffmask.sum()//10], hp1_cpu[i,ffmask].T[:,ffmask.sum()//10::ffmask.sum()//10]):
                R01 = rott[0].T
                t10 = -R01.dot(rott[1][:, 0])
        if not 't10' in locals():
            t10 = np.asarray([0, 0, 1])
            R01 = np.eye(3)
        rot[i] = torch.Tensor(cv2.Rodrigues(R01)[0][:, 0]).cuda(hp0.device)
        trans[i] = torch.Tensor(t10).cuda(hp0.device)

    return rot, trans, Es


def testEss(K0, K1, R, T, p1, p2):
    import cv2
    testP = cv2.triangulatePoints(K0.dot(np.concatenate((np.eye(3), np.zeros((3, 1))), -1)),
                                  K1.dot(np.concatenate((R, T), -1)),
                                  p1[:2], p2[:2])
    Z1 = testP[2, :] / testP[-1, :]
    Z2 = (R.dot(Z1 * np.linalg.inv(K0).dot(p1)) + T)[-1, :]
    if ((Z1 > 0).sum() > (Z1 <= 0).sum()) and ((Z2 > 0).sum() > (Z2 <= 0).sum()):
        # print(Z1)
        # print(Z2)
        return True
    else:
        return False
