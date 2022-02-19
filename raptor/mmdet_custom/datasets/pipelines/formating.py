from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formating import to_tensor


@PIPELINES.register_module()
class ImageToTensor_Custom:
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            if key not in results:
                if key not in results['img_info']:
                    continue
                else:
                    results[key] = to_tensor(results['img_info'][key])
            else:
                img = results[key]
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                results[key] = to_tensor(img.transpose(2, 0, 1))
            # img = results[key]
            # if len(img.shape) < 3:
            #     img = np.expand_dims(img, -1)
            # results[key] = (to_tensor(img.transpose(2, 0, 1))).contiguous()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class DefaultFormatBundle_Custom:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
    """

    def __init__(self, img_to_float=True):
        self.img_to_float = img_to_float

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img_fields' not in results:
            if 'img' in results:
                img = results['img']
                if self.img_to_float is True and img.dtype == np.uint8:
                    # Normally, image is of uint8 type without normalization.
                    # At this time, it needs to be forced to be converted to
                    # flot32, otherwise the model training and inference
                    # will be wrong. Only used for YOLOX currently .
                    img = img.astype(np.float32)
                # add default meta keys
                results = self._add_default_meta_keys(results)
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=True)
        else:
            for img_name in results['img_fields']:
                if img_name in results:
                    img = results[img_name]
                    # add default meta keys
                    results = self._add_default_meta_keys(results, img_name=img_name)
                    if len(img.shape) < 3:
                        img = np.expand_dims(img, -1)
                    img = np.ascontiguousarray(img.transpose(2, 0, 1))
                    results[img_name] = DC(to_tensor(img), stack=True)

        additional_keys = ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'calib_position', 'calib_K', 'calib_baseline']
        for im in ['tp', 'tm']:
            for fn in range(1, 20):
                additional_keys.append(f'calib_K_{im}_{fn}')
                additional_keys.append(f'calib_position_{im}_{fn}')
        for fn in range(3):
            additional_keys.append(f'calib_K_{fn}')
            additional_keys.append(f'calib_position_{fn}')

        for key in additional_keys:
            if key in results:
                results[key] = DC(to_tensor(results[key]))
            elif key not in results['img_info']:
                continue
            else:
                results[key] = DC(to_tensor(results['img_info'][key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results

    def _add_default_meta_keys(self, results, img_name='img'):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results[img_name]
        endname = img_name.replace('img', '')
        results.setdefault('pad_shape' + endname, img.shape)
        results.setdefault('scale_factor' + endname, 1.0)
        #img = results['img']
        #results.setdefault('pad_shape', img.shape)
        #results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg'+endname,
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'


@PIPELINES.register_module()
class Collect_Custom:
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:

            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')``
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'filename2', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'mcv_file')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        #img_meta = {}
        #for key in self.meta_keys:
        #    img_meta[key] = results[key]
        img_meta = {key: results[key] for key in self.meta_keys if key in results}
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
