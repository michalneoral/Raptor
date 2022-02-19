# Copyright (c) 2021 Toyota Motor Europe
# Patent Pending. All rights reserved.
#
# Author: Michal Neoral, CMP FEE CTU Prague
# Contact: neoramic@fel.cvut.cz
#
# This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>

import logging
from collections import OrderedDict

import mmcv
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

from raptor.tools.wandb_tools import *
import cv2
import scipy
import scipy.optimize
import png
import warnings

@DATASETS.register_module()
class CocoBasedMotionSegmentationDataset(CocoDataset):

    CLASSES = ('moving_object',)
    SHOW_EVAL_SUBSET = list(range(10)) # if None upload all images
    SHORT_CLASS_NAME = 'CocoBasedMotionSegmentation'
    SHOW_EVAL_THRESHOLD = 0.3

    def __init__(self, *args, **kwargs):
        super(CocoBasedMotionSegmentationDataset, self).__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            for fn in range(3):
                if f'file_name_{fn}' in info:
                    info[f'filename_{fn}'] = info[f'file_name_{fn}']
            for tm in range(1,20):
                if f'file_name_tm_{tm}' in info:
                    info[f'filename_tm_{tm}'] = info[f'file_name_tm_{tm}']
                else:
                    break
            for tp in range(1,20):
                if f'file_name_tp_{tp}' in info:
                    info[f'filename_tp_{tp}'] = info[f'file_name_tp_{tp}']
                else:
                    break
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_short_name(self):
        return self.SHORT_CLASS_NAME

    def format_evaluation_outputs(self, eval_outputs):
        formated_outputs = OrderedDict()
        for k,v in eval_outputs.items():
            formated_outputs[self.get_short_name() + '/' + k] = v
        return formated_outputs

    def _read_png_python(self, filepath, dtype=None, channels=1):
        dtype = dtype if dtype is not None else np.uint8
        reader = png.Reader(filepath)
        pngdata = reader.read()
        px_array = np.array(list(map(dtype, pngdata[2])))
        if channels != 1:
            px_array = px_array.reshape(-1, np.int32(px_array.shape[1] // channels), channels)
            if channels == 4:
                px_array = px_array[:, :, 0:3]
        else:
            px_array = np.expand_dims(px_array, axis=2)
        return px_array

    def _read_flow_png_python(self, file_path):
        flow = self._read_png_python(file_path, channels=3, dtype=np.uint16)
        flow = flow.astype(np.float32)
        u, v, valid = flow[:, :, 0], flow[:, :, 1], flow[:, :, 2]
        u = (u - 2 ** 15) / 64.
        v = (v - 2 ** 15) / 64.
        flow = np.stack([u, v, valid], axis=2)
        return flow

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 wandb_init_kwargs=None,
                 step=None,
                 **kwargs):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        eval_results = super(CocoBasedMotionSegmentationDataset, self).evaluate(results,
                                                                                metric=metric,
                                                                                logger=logger,
                                                                                jsonfile_prefix=jsonfile_prefix,
                                                                                classwise=classwise,
                                                                                proposal_nums=proposal_nums,
                                                                                iou_thrs=iou_thrs,
                                                                                metric_items=metric_items)

        eval_results = self.format_evaluation_outputs(eval_results)

        commit=False
        if wandb.run is None and wandb_init_kwargs is not None:
            wandb.init(**wandb_init_kwargs)
            commit=True
        if wandb.run is not None:
            subset = self.SHOW_EVAL_SUBSET if self.SHOW_EVAL_SUBSET is not None else None
            imgs = get_list_images(results, self.coco, self.img_prefix, subset=subset, segm_threshold=self.SHOW_EVAL_THRESHOLD)
            wandb.log({self.SHORT_CLASS_NAME: imgs}, commit=commit, step=step)

        return eval_results


@DATASETS.register_module()
class MoonkaCocoBasedMotionSegmentationDataset(CocoBasedMotionSegmentationDataset):
    SHOW_EVAL_SUBSET = list(range(10)) # if None upload all images
    SHORT_CLASS_NAME = 'Moonka'

@DATASETS.register_module()
class DrivingCocoBasedMotionSegmentationDataset(CocoBasedMotionSegmentationDataset):
    SHOW_EVAL_SUBSET = list(range(10)) # if None upload all images
    SHORT_CLASS_NAME = 'Driving'

@DATASETS.register_module()
class FlyingThingsCocoBasedMotionSegmentationDataset(CocoBasedMotionSegmentationDataset):
    SHOW_EVAL_SUBSET = list(range(10)) # if None upload all images
    SHORT_CLASS_NAME = 'FT3D'

@DATASETS.register_module()
class KittiCocoBasedMotionSegmentationDataset(CocoBasedMotionSegmentationDataset):
    SHOW_EVAL_SUBSET = list(range(30)) # if None upload all images
    SHORT_CLASS_NAME = 'KITTI'

    def __init__(self, *args, **kwargs):
        super(KittiCocoBasedMotionSegmentationDataset, self).__init__(*args, **kwargs)

    def read_flow(self, filepath):
        return self._read_flow_png_python(filepath)


    def evaluate(self,
                 results,
                 kitti_metric=None,
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 conf_thrs=None,
                 metric_items=None,
                 wandb_init_kwargs=None,
                 nproc=1,
                 step=None, **kwargs):

        eval_list_super = None
        eval_list_super = super(KittiCocoBasedMotionSegmentationDataset, self).evaluate(results,
                 metric=kwargs.get('metric', ['bbox']),
                 logger=logger,
                 jsonfile_prefix=jsonfile_prefix,
                 classwise=classwise,
                 proposal_nums=proposal_nums,
                 iou_thrs=iou_thrs,
                 metric_items=metric_items)

        self.nproc = 1 #nproc
        self.with_images = kwargs.get('with_images', True)

        with mmcv.Timer(print_tmpl='It tooks {}s to evaluate dataset '+self.SHORT_CLASS_NAME):
            if iou_thrs is None:
                iou_thrs = [0.01, 0.1, 0.3, 0.5, 0.75, 0.9, 0.95]
            if conf_thrs is None:
                conf_thrs = [0.3, 0.5, 0.7]

            imgs_idxs = self.coco.getImgIds()
            gt_idxs = []
            for img_idx in imgs_idxs:
                gt_idxs.append(self.coco.getAnnIds(img_idx))

            all_eval = self.get_eval(imgs_idxs, results, iou_thrs, conf_thrs, kitti_metric)
            eval_results = self.all_eval_dictlist2dict(all_eval)
            eval_results = self.format_evaluation_outputs(eval_results)

        commit = False
        if wandb.run is None and wandb_init_kwargs is not None:
            wandb.init(**wandb_init_kwargs)
            commit = True
        if wandb.run is not None:
            subset = self.SHOW_EVAL_SUBSET if self.SHOW_EVAL_SUBSET is not None else None
            if self.with_images:
                imgs = get_list_images(results, self.coco, self.img_prefix, subset=subset, segm_threshold=self.SHOW_EVAL_THRESHOLD)
                wandb.log({self.SHORT_CLASS_NAME: imgs}, commit=commit, step=step)

        if eval_list_super is not None:
            eval_results.update(eval_list_super)
        return eval_results

    def all_eval_dictlist2dict(self, all_eval):
        eval_results = {}
        eval_results_dict_list = {}
        for k, v in all_eval[0].items():
            for kk, vv in v.items():
                eval_results_dict_list[kk + '_{:03d}'.format(int(100 * k))] = []

        for c_eval in all_eval:
            for k, v in c_eval.items():
                for kk, vv in v.items():
                    eval_results_dict_list[kk + '_{:03d}'.format(int(100 * k))].append(vv)

        for k, v in eval_results_dict_list.items():
            if 'num' in k:
                c_value = np.nansum(v)
                c_value = c_value if not np.isnan(c_value) else 0.
                eval_results[k] = c_value
            else:
                c_value = np.nanmean(v)
                c_value = c_value if not np.isnan(c_value) else 0.
                eval_results[k] = c_value
        return eval_results

    def get_eval(self, imgs_idxs, results, iou_thrs, conf_thrs, kitti_metric):
        to_eval_list = []
        for idx, img_idx in enumerate(imgs_idxs):
            if 'flow_file' in self.coco.imgs[img_idx]:
                c_flowpath = osp.join(self.img_prefix, self.coco.imgs[img_idx]['flow_file'])
            else:
                c_flowpath = None

            if 'validation_mask_file' in self.coco.imgs[img_idx]:
                c_validpath = osp.join(self.img_prefix, self.coco.imgs[img_idx]['validation_mask_file'])
            else:
                c_validpath = None

            c_instancepath = osp.join(self.img_prefix, self.coco.imgs[img_idx]['inst_file'])
            c_results_segm = results[idx][1][0]
            c_results_bbox = results[idx][0][0]

            to_eval_list.append([c_flowpath, c_instancepath, c_results_segm, c_results_bbox, iou_thrs, conf_thrs, kitti_metric, c_validpath])

        if self.nproc > 1:
            all_eval = mmcv.track_parallel_progress(
                self.single_eval, to_eval_list, nproc=self.nproc)
        else:
            all_eval = mmcv.track_progress(self.single_eval, to_eval_list)

        return all_eval

    def single_eval(self, to_eval_list):
        warnings.filterwarnings("ignore")
        c_flowpath, c_instancepath, c_results_segm, c_results_bbox, iou_thrs, conf_thrs, kitti_metric, c_validpath = to_eval_list

        mask_gt = cv2.imread(c_instancepath, 0)
        if c_flowpath is not None:
            flow_gt = self.read_flow(c_flowpath).astype(np.float32)
            validmask = flow_gt[:, :, -1] == 1
            bgmask_gt = np.logical_and(validmask, mask_gt == 0).astype(float)
        elif c_validpath is not None:
            validmask_load = self._read_png_python(c_validpath, dtype=np.uint8, channels=1).astype(np.uint8)
            validmask = validmask_load[:,:,0] != 255
            mask_gt[mask_gt==255] = 1
            bgmask_gt = np.logical_and(validmask, mask_gt == 0).astype(float)
        else:
            bgmask_gt = (mask_gt == 0).astype(float)
            validmask = bgmask_gt >= 0

        c_eval_dict = {}
        for conf_threshold in conf_thrs:
            mask_pred = segmentation_single_image(c_results_segm, c_results_bbox, threshold=conf_threshold)
            if mask_pred is None:
                mask_pred = np.zeros_like(bgmask_gt)
            bgmask_pred = mask_pred == 0
            c_eval_dict[conf_threshold] = self.compute_iou(validmask, bgmask_pred, bgmask_gt, mask_pred, mask_gt, metrics=kitti_metric, iou_thrs=iou_thrs)

        return c_eval_dict

    def best_mask_fit(self, mask_pred, mask_gt, validmask=None):
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

    def compute_fbms_PRF(self, mask_gt, mask_est, idxs_gt, idxs_est):
        P = []
        R = []
        F = []
        if len(idxs_gt) == 0 or len(idxs_est) == 0:
            return 0, 0, 0
        for i in range(len(idxs_gt)):
            inter_size = np.logical_and(mask_gt == idxs_gt[i], mask_est == idxs_est[i]).sum()
            gt_size = np.sum(mask_gt == idxs_gt[i])
            est_size = np.sum(mask_est == idxs_est[i])
            P_c = inter_size / est_size
            R_c = inter_size / gt_size
            F_c = 2 * P_c * R_c / (P_c + R_c)
            if np.isnan(F_c):
                F_c = 0.
            P.append(P_c)
            R.append(R_c)
            F.append(F_c)
        return np.nanmean(P), np.nanmean(R), np.nanmean(F)

    def compute_towards_PRF(self, mask_gt, mask_est, idxs_gt, idxs_est, validmask=None):
        inter_size_cum = 0.
        gt_size_cum = np.sum(mask_gt > 0)
        est_size_cum = np.sum(mask_est > 0)
        for i in range(len(idxs_gt)):
            inter_size_cum += np.logical_and(mask_gt == idxs_gt[i], mask_est == idxs_est[i]).sum()

        # opposite dirrection for towards
        idxs_est_tow, idxs_gt_tow, _, _ = self.best_mask_fit(mask_gt, mask_est, validmask=validmask)
        for i in range(len(idxs_gt_tow)):
            if idxs_est_tow[i] in idxs_est:
                continue
            inter_size_cum += np.logical_and(mask_gt == idxs_gt_tow[i], mask_est == idxs_est_tow[i]).sum()

        P = inter_size_cum / est_size_cum
        R = inter_size_cum / gt_size_cum
        F = 2 * P * R / (P + R)
        P = P if not np.isnan(P) else 0.
        R = R if not np.isnan(R) else 0.
        F = F if not np.isnan(F) else 0.
        return P, R, F

    def compute_iou(self, validmask, bgmask_pred, bgmask_gt, mask_pred=None, mask_gt=None, metrics=None, iou_thrs=None):
        if iou_thrs is None:
            iou_thrs = [0.5]

        o = {}
        if 'bg' in metrics:
            bgiou = np.logical_and(bgmask_gt.astype(bool)[validmask], bgmask_pred[validmask]).sum() \
                    / np.logical_or(bgmask_gt.astype(bool)[validmask], bgmask_pred[validmask]).sum()
            o['bgiou'] = bgiou

        if 'obj' in metrics:
            idxs_gt, idxs_pred, o['objf'], computed_iou= self.best_mask_fit(mask_pred, mask_gt, validmask=validmask)
            o['P'], o['R'], o['F'] = self.compute_fbms_PRF(mask_gt, mask_pred, idxs_gt, idxs_pred)
            o['Pt'], o['Rt'], o['Ft'] = self.compute_towards_PRF(mask_gt, mask_pred, idxs_gt, idxs_pred, validmask=validmask)
            for iou_threshold in iou_thrs:
                o['num_pred_all_iou_{:03d}'.format(int(iou_threshold * 100))], \
                o['num_gt_all_iou_{:03d}'.format(int(iou_threshold * 100))], \
                o['num_pred_iou_{:03d}'.format(int(iou_threshold * 100))], \
                o['num_gt_iou_{:03d}'.format(int(iou_threshold * 100))], \
                o['num_FP0_iou_{:03d}'.format(int(iou_threshold * 100))], \
                o['num_FN0_iou_{:03d}'.format(int(iou_threshold * 100))] = \
                    self.compute_object_numbers_FPFN(mask_pred, mask_gt, idxs_gt, idxs_pred, validmask=validmask, computed_iou=computed_iou, iou_threshold=iou_threshold)
            o['average_overlap'] = self.compute_average_overlap(mask_pred, mask_gt, idxs_gt, idxs_pred, validmask=validmask)
            o['num_bg_gt'], o['num_bg_intersection'], o['num_bg_union'] = self.compute_background_precision(mask_pred, mask_gt, validmask=validmask)
        return o

    def compute_background_precision(self, mask_pred, mask_gt, validmask=None):
        bg_pred = mask_pred == 0
        bg_gt = mask_gt == 0
        if validmask is not None:
            bg_gt = bg_gt[validmask]
            bg_pred = bg_pred[validmask]
        num_bg_pred = np.sum(bg_pred)
        num_bg_gt = np.sum(bg_gt)
        num_bg_intersection = np.sum(np.logical_and(bg_pred, bg_gt))
        num_bg_union = np.sum(np.logical_or(bg_pred, bg_gt))
        return num_bg_gt, num_bg_intersection, num_bg_union

    def compute_object_numbers_FPFN(self, mask_pred, mask_gt, idxs_gt, idxs_pred, validmask=None, iou_threshold=None, computed_iou=None):
        num_pred_all = np.sum(np.unique(mask_pred) > 0)
        num_gt_all = np.sum(np.logical_and(np.unique(mask_gt) > 0, np.unique(mask_gt) != 255))
        #num_pred = len(idxs_pred)
        num_pred = np.sum(computed_iou >= iou_threshold)
        num_gt = len(idxs_gt)
        num_FN0 = num_gt_all - num_pred
        num_FP0 = num_pred_all - num_pred
        return num_pred_all, num_gt_all, num_pred, num_gt, num_FP0, num_FN0

    def compute_average_overlap(self, mask_pred, mask_gt, idxs_gt, idxs_pred, validmask=None):
        overlap = []
        num_gt = len(idxs_gt)
        for i in range(num_gt):
            idx_gt = idxs_gt[i]
            idx_pred = idxs_pred[i]
            obj_mask_gt = mask_gt == idx_gt
            obj_mask_pred = mask_pred == idx_pred

            if validmask is not None:
                obj_mask_gt = obj_mask_gt[validmask]
                obj_mask_pred = obj_mask_pred[validmask]

            area_gt = np.sum(obj_mask_gt)
            area_pred = np.sum(obj_mask_pred)
            area_all = np.sum(np.logical_or(obj_mask_pred, obj_mask_gt))
            area_inter = np.sum(np.logical_and(obj_mask_gt, obj_mask_pred))
            overlap.append(area_inter.astype(np.float)/area_all.astype(np.float))
        return np.nanmean(overlap)

@DATASETS.register_module()
class DavisCocoBasedMotionSegmentationDataset(KittiCocoBasedMotionSegmentationDataset):
    SHOW_EVAL_SUBSET = None # if None upload all images
    SHORT_CLASS_NAME = 'DAVIS'

@DATASETS.register_module()
class YtvosCocoBasedMotionSegmentationDataset(KittiCocoBasedMotionSegmentationDataset):
    SHOW_EVAL_SUBSET = None # if None upload all images
    SHORT_CLASS_NAME = 'YTVOS'

@DATASETS.register_module()
class KittiNewCocoBasedMotionSegmentationDataset(KittiCocoBasedMotionSegmentationDataset):
    SHOW_EVAL_SUBSET = None # if None upload all images
    SHORT_CLASS_NAME = 'KITTI_NEW'