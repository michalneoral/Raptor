# Copyright (c) 2021 Toyota Motor Europe
# Patent Pending. All rights reserved.
#
# Author: Michal Neoral, CMP FEE CTU Prague
# Contact: neoramic@fel.cvut.cz
#
# This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>

from mmcv import Config
import wandb
import imageio
import pycocotools.mask as maskUtils
import numpy as np
import os.path as osp
import png

def prepare_cfg_for_wandb(cfg, debug=False):
    init_kwargs = None
    if cfg.log_config is not None and cfg.log_config.hooks is not None:
        for hook in cfg.log_config.hooks:
            if hook.type is not None and hook.type == 'WandbLoggerHook':
                if hook.init_kwargs is None:
                    hook['init_kwargs'] = dict()
                hook['init_kwargs']['config'] = recursion_cfg2dict(cfg)
                if debug:
                    hook['init_kwargs']['project'] = 'debug_' + hook['init_kwargs']['project']
                init_kwargs = hook['init_kwargs']
    return cfg, init_kwargs


def recursion_cfg2dict(cfg):
    if isinstance(cfg, list):
        if len(cfg) > 0 and (isinstance(cfg[0], list) or isinstance(cfg[0], dict) or isinstance(cfg[0], Config)):
            return {idx:recursion_cfg2dict(v) for idx, v in enumerate(cfg)}
        else:
            return [recursion_cfg2dict(a) for a in cfg]
    elif isinstance(cfg, tuple):
        if len(cfg) > 0 and (isinstance(cfg[0], list) or isinstance(cfg[0], dict) or isinstance(cfg[0], Config)):
            return {idx: recursion_cfg2dict(v) for idx, v in enumerate(cfg)}
        else:
            return tuple(recursion_cfg2dict(a) for a in cfg)
    elif isinstance(cfg, Config) or isinstance(cfg, dict):
        return {k:recursion_cfg2dict(v) for k,v in cfg.items()}
    else:
        return cfg


def bounding_boxes_with_classes(v_boxes_list):
    all_boxes = []
    # plot each bounding box for this image
    for class_id in range(len(v_boxes_list)):
        v_boxes = v_boxes_list[class_id]
        max_id = v_boxes.shape[0]
        for i in range(max_id):
            # get coordinates and labels
            box_data = {
                "position" : {
                    "minX" : int(v_boxes[i,0]),
                    "minY" : int(v_boxes[i,1]),
                    "maxX" : int(v_boxes[i,2]),
                    "maxY" : int(v_boxes[i,3])
                },
                "class_id" : class_id,
                # optionally caption each box with its class and score
                "box_caption" : "%.3f" % (100*v_boxes[i,-1]),
                "domain" : "pixel",
                "scores" : { "score" : int(100*v_boxes[i,-1]) },
            }
            all_boxes.append(box_data)

    return all_boxes


def bounding_boxes(v_boxes):
    all_boxes = []
    # plot each bounding box for this image
    max_id = v_boxes.shape[0]
    for i in range(max_id):
        # get coordinates and labels
        box_data = {
            "position" : {
                "minX" : int(v_boxes[i,0]),
                "minY" : int(v_boxes[i,1]),
                "maxX" : int(v_boxes[i,2]),
                "maxY" : int(v_boxes[i,3])
            },
            "class_id" : 1,
            # optionally caption each box with its class and score
            "box_caption" : "%.3f" % (100*v_boxes[i,-1]),
            "domain" : "pixel",
            "scores" : { "score" : int(100*v_boxes[i,-1]) },
        }
        all_boxes.append(box_data)

    return all_boxes, max_id

def bounding_boxes_gt(v_boxes, category=None):
    all_boxes = []
    # plot each bounding box for this image
    max_id = v_boxes.shape[0]
    for i in range(max_id):
        # get coordinates and labels
        if category is not None:
            class_id = category[i]
        else:
            class_id = 1
        box_data = {
            "position": {
                "minX": int(v_boxes[i, 0]),
                "minY": int(v_boxes[i, 1]),
                "maxX": int(v_boxes[i, 0] + v_boxes[i, 2]),
                "maxY": int(v_boxes[i, 1] + v_boxes[i, 3])
            },
            "class_id": class_id,
            # optionally caption each box with its class and score
            "box_caption": "gt",
            "domain": "pixel",
            "scores": {"score": int(100)},
        }
        all_boxes.append(box_data)

    return all_boxes, max_id

    # log to wandb: raw image, predictions, and dictionary of class labels for each class id


def segmentation_single_image(segm, bbox, threshold):
    if len(segm) <= 0 or len(bbox) <=0:
        return None
    mask = maskUtils.decode(segm)
    mask = mask.astype(float)
    n = bbox.shape[0]

    masks_list = []
    for i in range(n):
        score = bbox[i, -1]
        if score >= threshold:
            masks_list.append(mask[:,:,i] * score)

    zeros = np.zeros(mask.shape[:2], dtype=mask.dtype)
    if len(masks_list) >= 0:
        mask = np.stack([zeros, *masks_list], axis=2)
        return np.nanargmax(mask, axis=-1).astype(np.uint8)
    else:
        return zeros


def _read_png_python(filepath, dtype=None, channels=1):
    dtype = dtype if dtype is not None else np.uint8
    reader = png.Reader(filepath)
    pngdata = reader.read()
    px_array = np.array(list(map(dtype, pngdata[2])))
    if channels != 1:
        px_array = px_array.reshape(-1, np.int32(px_array.shape[1] // channels), channels)
        if channels == 4:
            px_array = px_array[:,:,0:3]
    else:
        px_array = np.expand_dims(px_array, axis=2)
    return px_array


def _read_flow_png_python(file_path):
    flow = _read_png_python(file_path, channels=3, dtype=np.uint16)
    flow = flow.astype(np.float32)
    u, v, valid = flow[:,:,0], flow[:,:,1], flow[:,:,2]
    u = (u - 2 ** 15) / 64.
    v = (v - 2 ** 15) / 64.
    flow = np.stack([u, v, valid], axis=2)
    return flow


def get_gt_annot(dataset, idxs, flowpath=None, validpath=None, keep_category=False):
    bbox_list = []
    segm_list = []
    category_list = []
    for idx in idxs:
        bbox_list.append(dataset.anns[idx]['bbox'])
        segm_list.append(maskUtils.decode(dataset.anns[idx]['segmentation']))
        category_list.append(dataset.anns[idx]['category_id'])

    if len(bbox_list) <= 0 or len(segm_list) <= 0:
        return None, None, 0

    bbox = np.array(bbox_list)
    bbox, max_id = bounding_boxes_gt(bbox, category_list)
    if keep_category:
        segm = np.stack([seg * category_list[i] for i, seg in enumerate(segm_list)], axis=2)
    else:
        segm = np.stack([seg * (i+1) for i, seg in enumerate(segm_list)], axis=2)
    segm = np.max(segm, axis=2)

    if flowpath is not None:
        flow = _read_flow_png_python(flowpath)
        valid = flow[:,:,2]
        segm[valid==0] = 255 # remove invalid pixels from segmentation
    elif validpath is not None:
        validmask_load = _read_png_python(validpath, dtype=np.uint8, channels=1).astype(np.uint8)
        valid = validmask_load[:, :, 0] != 255
        segm[valid == 0] = 255  # remove invalid pixels from segmentation

    return bbox, segm, max_id


def get_list_images(prediction_list, dataset, img_prefix, subset=None, segm_threshold=0.3, classes=None):
    if subset is None:
        subset = []
    else:
        prediction_list = [prediction_list[sidx] for sidx in subset]
    imgs_idxs = dataset.getImgIds(subset)
    gt_idxs = []
    for img_idx in imgs_idxs:
        gt_idxs.append(dataset.getAnnIds(img_idx))

    imgs = []
    for idx, img_idx in enumerate(imgs_idxs):
        if len(prediction_list[idx][1]) > 0:
            c_segm_pred = segmentation_single_image(prediction_list[idx][1][0], prediction_list[idx][0][0], threshold=segm_threshold)
        else:
            c_segm_pred = None

        if len(prediction_list[idx][0]) > 0:
            c_bbox_pred, max_idx_pred = bounding_boxes(prediction_list[idx][0][0])
        else:
            c_bbox_pred = None
            max_idx_pred = 0

        c_filename = dataset.imgs[img_idx]['filename']
        if 'flow_file' in dataset.imgs[img_idx]:
            c_flowpath = osp.join(img_prefix, dataset.imgs[img_idx]['flow_file'])
        else:
            c_flowpath = None

        if 'validation_mask_file' in dataset.imgs[img_idx]:
            c_validpath = osp.join(img_prefix, dataset.imgs[img_idx]['validation_mask_file'])
        else:
            c_validpath = None

        c_filepath = osp.join(img_prefix, c_filename)
        c_bbox_gt, c_segm_gt, max_idx_gt = get_gt_annot(dataset, gt_idxs[idx], flowpath=c_flowpath, validpath=c_validpath)
        max_id = np.max([max_idx_pred, max_idx_gt])
        imgs.append(segmentation_with_bb(c_filepath, c_segm_pred, c_bbox_pred, None, None, max_id, caption='EST '+c_filename))
        imgs.append(segmentation_with_bb(c_filepath, None, None, c_segm_gt, c_bbox_gt, max_id, caption='GT '+c_filename))
        #imgs.append(segmentation_with_bb(c_filepath, c_segm_pred, c_bbox_pred, c_segm_gt, c_bbox_gt, max_id, caption=c_filename))

    return imgs


def get_list_images_gwent(prediction_list, dataset, img_prefix, subset=None, segm_threshold=0.3, classes=None):
    if subset is None:
        subset = []
    else:
        prediction_list = [prediction_list[sidx] for sidx in subset]
    imgs_idxs = dataset.getImgIds(subset)
    gt_idxs = []
    for img_idx in imgs_idxs:
        gt_idxs.append(dataset.getAnnIds(img_idx))

    imgs = []
    for idx, img_idx in enumerate(imgs_idxs):
        c_bbox_pred = bounding_boxes_with_classes(prediction_list[idx])

        c_filename = dataset.imgs[img_idx]['filename']
        if 'flow_file' in dataset.imgs[img_idx]:
            c_flowpath = osp.join(img_prefix, dataset.imgs[img_idx]['flow_file'])
        else:
            c_flowpath = None
        c_filepath = osp.join(img_prefix, c_filename)
        c_bbox_gt, c_segm_gt, max_idx_gt = get_gt_annot(dataset, gt_idxs[idx], flowpath=c_flowpath, keep_category=True)
        max_id = None
        imgs.append(segmentation_with_bb(c_filepath, None, c_bbox_pred, None, None, max_id, caption='EST '+c_filename, classes=classes))
        imgs.append(segmentation_with_bb(c_filepath, None, None, c_segm_gt, c_bbox_gt, max_id, caption='GT '+c_filename, classes=classes))
        #imgs.append(segmentation_with_bb(c_filepath, c_segm_pred, c_bbox_pred, c_segm_gt, c_bbox_gt, max_id, caption=c_filename))

    return imgs



def segmentation_with_bb(filename, segm_pred, bbox_pred, segm_gt, bbox_gt, max_id, caption=None, classes=None):
    if classes is None:
        display_ids = {"static": 0, "moving object": 1, "non_valid": 255}
        class_id_to_label_bbox = {int(v): k for k, v in display_ids.items()}
        for i in range(1, max_id):
            display_ids["moving object {:d}".format(i + 1)] = i + 1
        # this is a revese map of the integer class id to the string class label
        class_id_to_label_segm = {int(v): k for k, v in display_ids.items()}
    else:
        display_ids = {name: idx for idx,name in enumerate(classes)}
        class_id_to_label_bbox = {int(v): k for k, v in display_ids.items()}
        class_id_to_label_segm = {int(v)+1: k for k, v in display_ids.items()}

    boxes = dict()
    masks = dict()
    if bbox_pred is not None:
        boxes["predictions"] = {"box_data": bbox_pred,
                                "class_labels": class_id_to_label_bbox,}
    if bbox_gt is not None:
        boxes["ground_truth"] = {"box_data": bbox_gt,
                                 "class_labels": class_id_to_label_bbox,}

    if segm_pred is not None:
        masks["predictions"] = {"class_labels": class_id_to_label_segm,
                                "mask_data": segm_pred,}
    if segm_gt is not None:
        masks["ground_truth"] = {"class_labels": class_id_to_label_segm,
                                 "mask_data": segm_gt,}

    filename_jpg = filename.replace('.png', '.jpg')
    if not osp.isfile(filename_jpg):
        img = imageio.imread(filename)
        imageio.imwrite(filename_jpg, img, quality=75)

    return wandb.Image(filename_jpg,
                       caption=caption,
                       boxes=boxes,
                       masks=masks)



