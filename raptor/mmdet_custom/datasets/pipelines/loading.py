import os.path as osp

import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadTwoImagesFromFiles(object):
    """Load TWO image from files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    SAME FOR SECOND IMAGE WITH PATTERN "filename2" etc.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        results['img_fields'] = []

        name_group = [['filename', 'img']]
        name_group += [[f'filename_tp_{tp}', f'img_tp_{tp}'] for tp in range(1, 20)]
        name_group += [[f'filename_tm_{tm}', f'img_tm_{tm}'] for tm in range(1, 20)]
        name_group += [[f'filename{fn}', f'img{fn}'] for fn in range(3)]
        name_group += [[f'filename_{fn}', f'img{fn}'] for fn in range(3)]

        for names in name_group:
            filename_name = names[0]
            img_name = names[1]
            if filename_name not in results['img_info']:
                continue
            if results['img_prefix'] is not None:
                filename = osp.join(results['img_prefix'],
                                    results['img_info'][filename_name])
            else:
                filename = results['img_info'][filename_name]

            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)

            results[filename_name] = filename
            results['ori_'+filename_name] = results['img_info'][filename_name]
            results[img_name] = img
            results[img_name+'_shape'] = img.shape
            results['ori_shape'] = img.shape
            results['img_fields'].append(img_name)
            if 'mcv_file' in results['img_info']:
                results['mcv_file'] = results['img_info']['mcv_file']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
