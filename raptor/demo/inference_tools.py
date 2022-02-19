from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from raptor.mmdet_custom.apis import inference_moseg_detector
import mmcv
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import matplotlib.cm as cm
import imageio
from copy import deepcopy
import sys
import torch
# print(torch.cuda.is_available())
# print('torch: ',torch.version)
# print(torch.__version__)
# print('torch cuda: ',torch.version.cuda)
# print(torch.backends.cudnn.version())
from raptor.tools.wandb_tools import *
import wandb

import scipy
import scipy.optimize

import argparse
import datetime
import timeit


class Timer:
    """Measure time used."""
    # Ref: https://stackoverflow.com/a/57931660/

    def __init__(self, round_ndigits: int = 0):
        self._round_ndigits = round_ndigits
        self._start_time = timeit.default_timer()
        self._last_iter = self._start_time

    def __call__(self) -> float:
        return timeit.default_timer() - self._start_time

    def __str__(self) -> str:
        return str(datetime.timedelta(seconds=round(self(), self._round_ndigits)))

    def iter(self) -> float:
        last_iter = self._last_iter
        self._last_iter = timeit.default_timer()
        return  self._last_iter - last_iter

    def restart(self):
        self._last_iter = timeit.default_timer()


def flow_torch2numpy_resize_rescale(flow_torch, dimensions):

    flow_numpy = flow_torch.detach().cpu().numpy()[0,:,:,:].transpose([1,2,0])
    dims_flow = flow_numpy.shape
    flow_numpy = cv2.resize(flow_numpy, dsize=(dimensions[1], dimensions[0]), interpolation=cv2.INTER_LINEAR)
    flow_numpy[:,:,0] = flow_numpy[:,:,0] * (dimensions[1] / dims_flow[1])
    flow_numpy[:,:,1] = flow_numpy[:,:,1] * (dimensions[0] / dims_flow[0])

    return flow_numpy


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def bw_bilinear_interpolate_flow(im, flow):
    ndim = im.ndim
    if ndim == 2:
        im = np.expand_dims(im, axis=2)
    H, W, C = im.shape
    X_g, Y_g = np.meshgrid(range(W), range(H))
    x, y = flow[:, :, 0], flow[:, :, 1]
    x = x + X_g
    y = y + Y_g
    im_w = []
    for i in range(C):
        im_w.append(bilinear_interpolate(im[:, :, i], x, y))
    im_w = np.stack(im_w, axis=2)

    if ndim == 2:
        im_w = im_w[:, :, 0]
    return im_w


def closest_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    tmp = Ia == Ib
    wa[tmp] += wb[tmp]

    tmp = Ia == Ic
    wa[tmp] += wc[tmp]

    tmp = Ia == Id
    wa[tmp] += wd[tmp]

    tmp = Ib == Ic
    wb[tmp] += wc[tmp]

    tmp = Ib == Id
    wb[tmp] += wd[tmp]

    tmp = Ic == Id
    wc[tmp] += wd[tmp]

    argmax = np.argmax(np.stack([wa, wb, wc, wd], axis=2), axis=2)
    intepImage = np.choose(argmax, [Ia, Ib, Ic, Id])
    return intepImage


def bw_closest_interpolate_flow(im, flow):
    ndim = im.ndim
    if ndim == 2:
        im = np.expand_dims(im, axis=2)
    H, W, C = im.shape
    X_g, Y_g = np.meshgrid(range(W), range(H))
    x, y = flow[:, :, 0], flow[:, :, 1]
    x = x + X_g
    y = y + Y_g
    im_w = []
    for i in range(C):
        im_w.append(closest_interpolate(im[:, :, i], x, y))
    im_w = np.stack(im_w, axis=2)

    if ndim == 2:
        im_w = im_w[:, :, 0]
    return im_w


def bw_closest_interpolate_sep_masks_flow(sep_masks, flow):
    sep_masks_interp = []
    for i in range(len(sep_masks)):
        sep_masks_interp.append((sep_masks[i][0], bw_closest_interpolate_flow(sep_masks[i][1], flow)))
    return sep_masks_interp


def best_mask_sep_masks_fit(masks_pred, masks_gt, validmask=None):
    if validmask is None:
        validmask = np.ones_like(masks_pred[0][1]) > 0
    gtlist = range(len(masks_gt))
    estlist = range(len(masks_pred))
    M = len(gtlist)
    PR = len(estlist)
    imatrix = np.zeros((M, PR))
    fmatrix = np.zeros((M, PR))
    for i in range(M):  # for all bg instances
        objx_mask_gt = masks_gt[i][1] > 0
        for j in range(PR):
            objx_mask_pred = masks_pred[j][1] > 0
            imatrix[i, j] = float(np.logical_and(objx_mask_gt, objx_mask_pred)[validmask].sum())
            fmatrix[i, j] = imatrix[i, j] / (objx_mask_gt[validmask].sum() + objx_mask_pred[validmask].sum()) * 2
    fmatrix[np.isnan(fmatrix)] = 0.
    fmatrix[np.isinf(fmatrix)] = 0.
    ridx, cidx = scipy.optimize.linear_sum_assignment(1 - fmatrix)
    # objf = imatrix[ridx, cidx].sum() / ((mask_pred > 0)[validmask].sum() + (mask_gt > 0)[validmask].sum()) * 2
    idxs_gt = [gtlist[i] for i in ridx]
    idxs_pred = [estlist[i] for i in cidx]
    return idxs_gt, idxs_pred


def best_mask_fit(mask_pred, mask_gt, validmask=None):
    if validmask is None:
        validmask = np.ones_like(mask_pred) > 0
    gtlist = list(set(mask_gt.flatten()))
    M = len(gtlist) - 1
    estlist = list(set(mask_pred.flatten()))
    PR = len(estlist) - 1
    imatrix = np.zeros((M, PR))
    fmatrix = np.zeros((M, PR))
    for i in range(M):  # for all bg instances
        objx_mask_gt = mask_gt == gtlist[i + 1]
        for j in range(PR):
            objx_mask_pred = mask_pred == estlist[j + 1]
            imatrix[i, j] = float(np.logical_and(objx_mask_gt, objx_mask_pred)[validmask].sum())
            fmatrix[i, j] = imatrix[i, j] / (objx_mask_gt[validmask].sum() + objx_mask_pred[validmask].sum()) * 2
    fmatrix[np.isnan(fmatrix)] = 0.
    fmatrix[np.isinf(fmatrix)] = 0.
    ridx, cidx = scipy.optimize.linear_sum_assignment(1 - fmatrix)
    # objf = imatrix[ridx, cidx].sum() / ((mask_pred > 0)[validmask].sum() + (mask_gt > 0)[validmask].sum()) * 2
    idxs_gt = [gtlist[i + 1] for i in ridx]
    idxs_pred = [estlist[i + 1] for i in cidx]
    objf = imatrix[ridx, cidx].sum() / ((mask_pred > 0)[validmask].sum() + (mask_gt > 0)[validmask].sum()) * 2
    return idxs_gt, idxs_pred, objf, fmatrix[ridx, cidx]

def reassamble_ids(mask, old_ids, new_ids):
    if len(new_ids) == 0:
        return mask
    mask_new = np.zeros_like(mask)
    all_old_ids = np.unique(mask)
    free_all_old_ids_wout_zero = list(set(all_old_ids) - set([0]) - set(old_ids))
    free_all_old_ids_wout_zero.sort()

    max_id = np.max([np.max(all_old_ids), np.max(old_ids), np.max(new_ids)])
    free_ids = set(range(1,max_id+1))
    free_ids = list(free_ids - set(new_ids))
    free_ids.sort()

    np.delete(all_old_ids, 0)
    for i in range(len(old_ids)):
        mask_new[mask == old_ids[i]] = new_ids[i]

    for i in range(len(free_all_old_ids_wout_zero)):
        mask_new[mask == free_all_old_ids_wout_zero[i]] = free_ids[i]
    return mask_new


def reassamble_sep_masks_ids(sep_masks, old_ids, new_ids):
    if len(new_ids) == 0:
        return sep_masks
    sep_masks_reid = [None] * np.max([len(sep_masks)+1, np.max(old_ids)+1, np.max(new_ids)+1])
    for i in range(len(sep_masks_reid)):
        sep_masks_reid[i] = (0.0, np.zeros_like(sep_masks[0][1]))

    all_old_ids = range(len(sep_masks))
    free_all_old_ids = list(set(all_old_ids) - set(old_ids))
    free_all_old_ids.sort()

    max_id = np.max([np.max(all_old_ids), np.max(old_ids), np.max(new_ids)])
    free_ids = set(range(1, max_id + 1))
    free_ids = list(free_ids - set(new_ids))
    free_ids.sort()

    for i in range(len(old_ids)):
        sep_masks_reid[new_ids[i]] = sep_masks[old_ids[i]]

    for i in range(len(free_all_old_ids)):
        sep_masks_reid[free_ids[i]] = sep_masks[free_all_old_ids[i]]

    return sep_masks_reid


def simple_tracker(flow, sep_masks, label_mask, sep_masks_prev, label_mask_prev):
    if len(sep_masks) == 0 or len(sep_masks_prev) == 0:
        return sep_masks, label_mask
    label_mask_prev_interp = bw_closest_interpolate_flow(label_mask_prev, flow)
    id_prev, id_current, objf, fmatrix = best_mask_fit(label_mask, label_mask_prev_interp)
    label_mask_reid = reassamble_ids(label_mask, old_ids=id_current, new_ids=id_prev)

    sep_masks_prev_interp = bw_closest_interpolate_sep_masks_flow(sep_masks_prev, flow)
    id_prev, id_current = best_mask_sep_masks_fit(sep_masks, sep_masks_prev_interp)
    sep_masks_reid = reassamble_sep_masks_ids(sep_masks, old_ids=id_current, new_ids=id_prev)

    return sep_masks_reid, label_mask_reid

def getArgs(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='input arguments for evaluation')
    parser.add_argument('--gpuid', default='1')
    parser.add_argument('--config_file', default=None)
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--type', default='clean')
    parser.add_argument('--bwfw', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_outputs', action='store_true')
    parser.add_argument('--save_custom_outputs', action='store_true')

    return parser.parse_args(argv)

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def load_calib_kitti(cam_to_cam_file):
    # We'll return the camera calibration as a dictionary
    data = {}

    # Load and parse the cam-to-cam calibration data
    filedata = read_calib_file(cam_to_cam_file)

    # Create 3x4 projection matrices
    P_rect_20 = np.reshape(filedata['P2'], (3, 4))
    P_rect_30 = np.reshape(filedata['P3'], (3, 4))

    # Compute the camera intrinsics
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    data['b20'] = P_rect_20[0, 3] / P_rect_20[0, 0]
    data['b30'] = P_rect_30[0, 3] / P_rect_30[0, 0]

    return data

def filter_checkpoint_file(filename):
    pretrained_dict_mot = torch.load(filename, map_location='cpu')

    state_dict = pretrained_dict_mot['state_dict']
    pretrained_dict_mot['state_dict'] = dict()
    for k, v in state_dict.items():
        if ('reg_modules' not in k or 'conv1' in k) and ('grid' not in k) and ('flow_reg' not in k) and ('midas' not in k):
            pretrained_dict_mot['state_dict'][k] = v

    filename2 = filename.replace('.pth', '_tmp.pth')
    torch.save(pretrained_dict_mot, filename2)
    return filename2

def load_and_change_config(config, size):
    config = mmcv.Config.fromfile(config)
    def recursive_change_size(config, size):
        if isinstance(config, dict) or isinstance(config, mmcv.utils.config.Config):
            for c in config.keys():
                if c == 'img_scale':
                    config[c] = (size[0], size[1])
                # elif c == 'samples_per_gpu':
                #     config[c] = 1
                else:
                    config[c] = recursive_change_size(config[c], size)
        if isinstance(config, list):
            for ci in range(len(config)):
                config[ci] = recursive_change_size(config[ci], size)
        return config
    return recursive_change_size(config, size)

def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def mkdir_from_full_file_path_if_not_exist(path):
    basename = os.path.basename(path)
    mkdir_if_not_exist(path[:-len(basename)])

def save_labels(labels, img, save_path=None):
    alpha = 0.7
    additional_text = 'lmax = {:d}'.format(labels.max())

    color_labels = cm.jet(labels / labels.max())
    color_img_labels = img * (1 - alpha) + color_labels[:, :, :3] * alpha
    edged = cv2.Canny(labels.astype(np.uint8), 0, 1)
    edged_3 = np.stack([edged] * 3, axis=2)

    color_img_labels_edged = (color_img_labels * 255).astype(np.uint8) + edged_3

    if save_path is not None:
        save_path = save_path[:-4] + '.jpg'
        mkdir_from_full_file_path_if_not_exist(save_path)
        imageio.imwrite(save_path, color_img_labels_edged)
    else:
        plt.imshow(color_img_labels_edged)

def save_masks(sep_masks, save_path=None):
    for i in range(len(sep_masks)):
        conf, mask = sep_masks[i]

        if save_path is not None:
            save_path_c = deepcopy(save_path)
            save_path_c = save_path_c[:-4] + '_n_{:05d}_conf_{:.3f}.png'.format(i, conf)
            mkdir_from_full_file_path_if_not_exist(save_path_c)
            imageio.imwrite(save_path_c, mask)
        else:
            plt.imshow(mask)
            plt.show()

def save_single_mask(sep_masks, loaded_img, save_path=None, sort_masks=True):
    conf_list = []
    mask_list = []
    all_mask = np.zeros(loaded_img.shape[:2], dtype=np.uint16)

    if sort_masks:
        _, new_mask_list = sort_sep_masks(sep_masks)
    else:
        new_mask_list = [x for _, x in sep_masks]

    if new_mask_list is not None and len(new_mask_list) > 0:
        len_mask_list = len(new_mask_list)
        for i in range(len_mask_list):
            current_mask = new_mask_list[i]
            all_mask[current_mask>0] = len_mask_list - i

    if save_path is not None:
        save_path_c = deepcopy(save_path)
        save_path_c = save_path_c[:-4] + '.png'
        mkdir_from_full_file_path_if_not_exist(save_path_c)
        imageio.imwrite(save_path_c, all_mask)
    else:
        plt.imshow(all_mask)
        plt.show()

# %%

def create_mask(img_path, detection, threshold=0.3):
    result = detection
    n_objects = 0
    loaded_img = np.asarray(Image.open(img_path), float) / 255
    label_mask = np.zeros(loaded_img.shape[0:2], dtype=int)

    for idx in range(len(result[1])):
        m = result[1][idx]
        if len(m) > 0:
            for idx2, g in enumerate(m):
                if isinstance(g, dict):
                    g = maskUtils.decode(g) > 0
                r = result[0][idx][idx2]

                if r[-1] > threshold:
                    n_objects += 1
                    label_mask[g] = n_objects
    return loaded_img, label_mask

def mask_separation(img_path, detection, threshold=0.3):
    result = detection
    n_objects = 0
    loaded_img = np.asarray(Image.open(img_path), float) / 255
    label_mask = np.zeros(loaded_img.shape[0:2], dtype=int)

    separated_masks = []

    for idx in range(len(result[1])):
        m = result[1][idx]
        if len(m) > 0:
            for idx2, g in enumerate(m):
                if isinstance(g, dict):
                    g = maskUtils.decode(g) > 0 # ((maskUtils.decode(g) > 0) * 1).astype()
                r = result[0][idx][idx2]
                conf = r[-1]
                if conf >= threshold:
                    label_mask = np.zeros(loaded_img.shape[0:2], dtype=np.uint8)
                    n_objects += 1
                    label_mask[g] = 255
                    separated_masks.append((conf, label_mask))
    return separated_masks

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.

    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT, check)
    M = np.fromfile(f, dtype='float64', count=9).reshape((3, 3))
    N = np.fromfile(f, dtype='float64', count=12).reshape((3, 4))
    return M, N


def load_calib_sintel(test_left_img):
    seqname1 = test_left_img.split('/')[-2]
    framename = int(test_left_img.split('/')[-1].split('_')[-1].split('.')[0])
    K0, _ = cam_read('/datagrid/public_datasets/Sintel-complete/MPI-Sintel-depth-training-20150305/training/camdata_left/%s/frame_%04d.cam' % (seqname1, framename))
    bl = 0.1
    return K0, bl


def sort_sep_masks(sep_masks=None):
    conf_list = []
    mask_list = []

    new_mask_list = None
    new_sep_masks = None

    if sep_masks is not None:
        new_sep_masks = []
        for i in range(len(sep_masks)):
            conf, mask = sep_masks[i]
            conf_list.append(conf)
            mask_list.append(mask)

        if len(sep_masks) > 0:
            try:
                new_mask_list = [x for _, x in sorted(zip(conf_list, mask_list))]
                new_sep_masks = [(c,x) for c, x in sorted(zip(conf_list, mask_list))]
            except:
                conf_list = [1.0 - 0.01 * i for i in range(len(conf_list))]
                new_mask_list = [x for _, x in sorted(zip(conf_list, mask_list))]
                new_sep_masks = [(c, x) for c, x in sorted(zip(conf_list, mask_list))]
    return new_sep_masks, new_mask_list


def custom_image(loaded_img, sep_masks=None, labels_gt=None, contour=True, contour_color='w', active=True, figsize=(19.8, 3), interpolation='nearest', save_path=None,
                      show_image=True, sort_masks=True):
    # plt.figure(figsize=figsize)

    all_mask = np.zeros(loaded_img.shape[:2], dtype=np.uint16)

    if sort_masks:
        _, new_mask_list = sort_sep_masks(sep_masks)
    else:
        new_mask_list = [x for _, x in sep_masks]

    if sort_masks:
        if new_mask_list is not None:
            if len(new_mask_list) > 0:
                len_mask_list = len(new_mask_list)
                for i in range(len_mask_list):
                    current_mask = new_mask_list[i]
                    all_mask[current_mask > 0] = len_mask_list - i
    else:
        if new_mask_list is not None:
            if len(new_mask_list) > 0:
                len_mask_list = len(new_mask_list)
                for i in range(len_mask_list):
                    current_mask = new_mask_list[i]
                    all_mask[current_mask > 0] = i + 1
                all_mask[0,0] = ((np.floor(len_mask_list / 10) + 1) * 10) - 1

    if sep_masks is not None:
        save_custom_image(loaded_img, all_mask, labels_gt=labels_gt, contour=contour, contour_color=contour_color, active=active, figsize=figsize, interpolation=interpolation, save_path=save_path, show_image=show_image)


def save_custom_image(loaded_img, all_mask=None, labels_gt=None, contour=True, contour_color='w', active=True, figsize=(19.8, 3), interpolation='nearest', save_path=None,
                      show_image=True):
    fig = plt.figure(figsize=figsize, frameon=False)
    # fig.set_siz(img.shape[1]*2+5, img.shape[0])

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    dist_panel = np.ones([loaded_img.shape[0], 5, 3], dtype=np.uint8)

    if all_mask is None and labels_gt is not None:
        imgs = loaded_img
        labels = labels_gt
    elif labels_gt is not None:
        labels = np.concatenate([labels_gt, 10 * dist_panel[:, :, 0], all_mask], axis=1)
        imgs = np.concatenate([loaded_img, dist_panel * 255, loaded_img], axis=1)
    else:
        imgs = loaded_img
        labels = all_mask

    if labels is not None:
        # plt.imshow(imgs, 'gray', interpolation='none')
        # plt.imshow(labels, 'tab10', interpolation='none', alpha=0.7)

        ax.imshow(imgs, 'gray', interpolation='none', aspect='auto')
        ax.imshow(labels%10, 'tab10', interpolation='none', alpha=0.7, aspect='auto')

    if contour:
        # plt.contour(labels, colors=contour_color)

        ax.contour(labels, colors=contour_color, aspect='auto')

    plt.axis('image')
    plt.axis('off')

    # plt.tight_layout()

    if save_path is not None:
        save_path = save_path[:-4] + '.jpg'
        mkdir_from_full_file_path_if_not_exist(save_path)
        fig.savefig(save_path)

    if show_image:
        plt.show()

    else:
        plt.close(fig)
