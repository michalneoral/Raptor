from mmdet.models.builder import DETECTORS
from .cascade_rcnn_raptor import CascadeRCNNRaptor


@DETECTORS.register_module()
class HybridTaskCascadeRaptor(CascadeRCNNRaptor):
    """Implementation of `HTC <https://arxiv.org/abs/1901.07518>`_"""

    def __init__(self, **kwargs):
        super(HybridTaskCascadeRaptor, self).__init__(**kwargs)

    @property
    def with_semantic(self):
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic
