# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
MOTION_BACKBONES = MODELS
BACKBONES = MODELS

def build_motion_backbone(cfg):
    """Build motion backbone."""
    return MOTION_BACKBONES.build(cfg)

def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)