# Copyright (c) 2021 Toyota Motor Europe
# Patent Pending. All rights reserved.
#
# Author: Michal Neoral, CMP FEE CTU Prague
# Contact: neoramic@fel.cvut.cz
#
# This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class CocoDatasetClassAgnostic(CocoDataset):

    CLASSES = ('object')

