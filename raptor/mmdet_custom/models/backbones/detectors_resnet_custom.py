from mmdet.models.builder import BACKBONES
from mmdet.models.backbones import DetectoRS_ResNet
from mmdet.models.backbones.detectors_resnet import Bottleneck

@BACKBONES.register_module()
class DetectoRS_ResNet_Custom(DetectoRS_ResNet):
    """ResNet backbone for DetectoRS.

    Added ResNet18, ResNet34 - with bottlenecks instead of BasicBlock
    """

    arch_settings = {
        18: (Bottleneck, (2, 2, 2, 2)),
        34: (Bottleneck, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
