import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose

def inference_moseg_detector(model, img, img2, calib_K, calib_K_2=None, calib_baseline=1., img0=None, calib_K_0=None, additional_outputs_setting=None):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    if calib_K_2 is None:
        calib_K_2 = calib_K

    if img0 is not None:
        if calib_K_0 is None:
            calib_K_0 = calib_K

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # prepare data
    if isinstance(img, np.ndarray) and isinstance(img2, np.ndarray):
        # directly add img
        data = dict(img=img, img2=img2, calib_K=calib_K, calib_K_2=calib_K_2, calib_baseline=calib_baseline)
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadTwoImagesFromWebcam'
    elif not isinstance(img, np.ndarray) and not isinstance(img2, np.ndarray):
        # add information into dict
        if img0 is None:
            data = dict(img_info=dict(filename=img,
                                  filename2=img2,
                                  calib_K=calib_K,
                                  calib_K_2=calib_K_2,
                                  calib_baseline=calib_baseline
                                  ), img_prefix=None)
        else:
            data = dict(img_info=dict(filename=img,
                                  filename2=img2,
                                  filename0=img0,
                                  calib_K=calib_K,
                                  calib_K_2=calib_K_2,
                                  calib_K_0=calib_K_0,
                                  calib_baseline=calib_baseline
                                  ), img_prefix=None)
    else:
        raise ValueError('Bad combination of the input data')
    # build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    if additional_outputs_setting is not None:
        model.set_additional_outputs(additional_outputs_setting=additional_outputs_setting)

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)[0]
        if additional_outputs_setting is not None:
            additional_outputs = model.get_additional_outputs()
            return result, additional_outputs
    return result