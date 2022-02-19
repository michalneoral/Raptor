from mmdet.models.builder import DETECTORS
from .two_stage_raptor import TwoStageDetectorRaptor

@DETECTORS.register_module()
class CascadeRCNNRaptor(TwoStageDetectorRaptor):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone=None,
                 motion_backbone=None,
                 motion_backbone_before_backbone=None,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CascadeRCNNRaptor, self).__init__(
            backbone=backbone,
            motion_backbone=motion_backbone,
            motion_backbone_before_backbone=motion_backbone_before_backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector."""
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        elif isinstance(result, dict):
            result = result['ensemble']
        return super(CascadeRCNNRaptor, self).show_result(data, result, **kwargs)
