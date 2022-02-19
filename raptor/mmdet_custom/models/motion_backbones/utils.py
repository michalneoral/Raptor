from __future__ import print_function, division

import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path as osp

import re
from kornia.geometry.transform import resize as kornia_resize



import sys
class SetTrace(object):
    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        sys.settrace(None)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def show_image(img, n_image=0, channel=0, rgb=False, **kwargs):
    img = img.detach().cpu().numpy()

    if img.ndim == 4:
        img = img[n_image]
    if img.ndim == 3:
        n_ch = img.shape[0]

        if n_ch == 3 and rgb:
            img = img.transpose(1, 2, 0)
        else:
            img = img[channel]

    show_image_pyplot(img, **kwargs)

def readPFMpython3(filename):
    file = open(filename, 'rb')

    color = None
    width = None
    scale = None
    endian = None

    header = file.readline().decode('latin-1').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('latin-1'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('latin-1').rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def read_flow_pfm_numpy(file):
    data, scale = readPFMpython3(file)
    data.astype(np.float32)
    return data

def load_flow_ft3d(img_metas):
    f1 = img_metas['filename']
    f2 = img_metas['filename2']
    f1_idx = int(osp.basename(f1).split('.')[0])
    f2_idx = int(osp.basename(f2).split('.')[0])
    if f1_idx < f2_idx:
        prefix = 'into_future/'
        suffix = 'OpticalFlowIntoFuture_'
    else:
        prefix = 'into_past/'
        suffix = 'OpticalFlowIntoPast_'

    flow_path = f1.replace('frames_finalpass_png', 'optical_flow').replace('frames_cleanpass', 'optical_flow')
    flow_path = flow_path.replace('right/',prefix+'right/').replace('left/',prefix+'left/')
    if 'left' in flow_path:
        flow_path = flow_path.replace('.png', '_L.pfm')
    else:
        flow_path = flow_path.replace('.png', '_R.pfm')
    flow_path = osp.join(osp.dirname(flow_path), suffix+osp.basename(flow_path))
    flow = read_flow_pfm_numpy(flow_path)
    flow, valid = flow[:,:,:2], flow[:,:,2]
    return flow, valid

def load_debug_flows_ft3d(img_metas, cuda_device_name=None):
    B = len(img_metas)
    flow_np = []
    for i in range(B):
        c_flow, c_valid = load_flow_ft3d(img_metas[i])
        flow_np.append(c_flow)

    flow_np = np.stack(flow_np, axis=0)
    flow_np = flow_np.transpose(0, 3, 1, 2)

    pad_shape = img_metas[0]['pad_shape']
    ori_shape = img_metas[0]['ori_shape']
    flow_torch = torch.from_numpy(flow_np).cuda(cuda_device_name)
    flow_torch = kornia_resize(flow_torch, (pad_shape[0], pad_shape[1]), align_corners=True)

    flow_torch[:, 0, :, :] = flow_torch[:, 0, :, :] * pad_shape[1] / ori_shape[1]
    flow_torch[:, 1, :, :] = flow_torch[:, 1, :, :] * pad_shape[0] / ori_shape[0]

    return flow_torch

def show_flow(flow, clip_flow=None, clip_mag=None, **kwargs):
    flow = flow.detach().cpu().numpy()
    flow = flow.transpose(1,2,0)
    image = flow_to_color(flow, clip_flow=clip_flow, clip_mag=clip_mag)
    show_image_pyplot(image, **kwargs)


def show_image_pyplot(img, title='', contour=None, labels=None, colorbar=False, contour_color='w', active=True, figsize=None, interpolation='nearest'):
    if figsize is not None:
        plt.figure(figsize=figsize)
    if labels is not None:
        plt.imshow(img, 'gray', interpolation='none')
        plt.imshow(labels, 'jet', interpolation='none', alpha=0.7)
    else:
        plt.imshow(img, interpolation=interpolation)
    if colorbar:
        plt.colorbar()
    plt.title(title)
    if contour is not None:
        plt.contour(contour, colors=contour_color)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def flow_to_color(flow_uv, clip_flow=None, clip_mag=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]

    rad = np.sqrt(np.square(u) + np.square(v))
    if clip_mag is not None:
        rad_max = clip_mag
    else:
        rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False, clip_radius=None):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    if clip_radius is not None:
        rad = np.clip(rad, -clip_radius, clip_radius)

    a = np.arctan2(-v, -u)/np.pi

    fk = (a+1) / 2*(ncols-1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):

        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1

        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)

    return flow_image