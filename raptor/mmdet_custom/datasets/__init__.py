
from .coco_class_agnostic import CocoDatasetClassAgnostic
# from .ft3d_moseg import FlyingThingsMotionSegmentationDataset
from .coco_moseg import (
    CocoBasedMotionSegmentationDataset,
    FlyingThingsCocoBasedMotionSegmentationDataset,
    MoonkaCocoBasedMotionSegmentationDataset,
    DrivingCocoBasedMotionSegmentationDataset,
    KittiCocoBasedMotionSegmentationDataset,
    DavisCocoBasedMotionSegmentationDataset,
    KittiNewCocoBasedMotionSegmentationDataset,
    YtvosCocoBasedMotionSegmentationDataset,
)

__all__ = [
    'CocoDatasetClassAgnostic',
    'FlyingThingsCocoBasedMotionSegmentationDataset', 'MoonkaCocoBasedMotionSegmentationDataset',
    'DrivingCocoBasedMotionSegmentationDataset', 'CocoBasedMotionSegmentationDataset',
    'KittiCocoBasedMotionSegmentationDataset', 'DavisCocoBasedMotionSegmentationDataset',
    'KittiNewCocoBasedMotionSegmentationDataset', 'YtvosCocoBasedMotionSegmentationDataset',
]
