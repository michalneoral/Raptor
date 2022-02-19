# DATASETS

Neoral, M.; Šochman, J. & Matas, J.: "**Monocular Arbitrary Moving Object Discovery and Segmentation**", BMVC 2021<br>
For more details: [PAPER-PDF](https://www.bmvc2021-virtualconference.com/assets/papers/1500.pdf)

### Authors:
* Michal Neoral (neoramic@fel.cvut.cz)
* Jan Šochman (jan.sochman@fel.cvut.cz)
* Jiří Matas (matas@fel.cvut.cz, matas@cmp.felk.cvut.cz)

## KITTI'15 Dataset Instance Motion Segmentation Extension

We extended the official KITTI'15 dataset[^menze2015].
independently moving instance segmentation ground truth to cover all moving objects, not
just a selection of cars and vans.

### Download
```
cd YOUR_kitti15_basic_training_OR_multiview_training_directory
gdown https://drive.google.com/uc?id=1tOvHlFfL0cNIVJNjg2iC6JXyXmELoH4R -O ./KITTI15_MSplus.zip
unzip KITTI15_MSplus.zip
```

### Data Description
```
├── readme.txt
│   └── file with link to this repo
├── motion_segmentation_coco_format/
│   └── directory contating all evaluation COCO-format files for both KITTI'15 and KITTI'15 IMS Extension
├── obj_map_fg/
│   └── binary *.png maps for foreground/background segmentation - all moving instances as foreground
├── obj_map_moseg/
│   └── instance motion segmentation *.png maps
└── obj_map_valid/
    └── binary *.png masks of motion segmenation - motion edges and areas excluded by annotator 
```
      
## SceneFlow Dataset Instance Motion Segmentation Extension

The extension of SceneFlow Dataset[^mayer2016] - instance motion segmentation.

### Download
```
cd YOUR_sceneflowdataset_directory
gdown https://drive.google.com/uc?id=1eopYfpy8Ru7PpsemuVDyNXjj9XqDCZKm -O ./SceneFlow_MSplus.zip
unzip SceneFlow_MSplus.zip
```


### Data Description
```
├── readme.txt
│   └── file with link to this repo
├── motion_segmentation_coco_format/
│   └── directory contating all evaluation COCO-format files for both KITTI'15 and KITTI'15 IMS Extension
├── binary_motion_segmentation_png/
│   └── binary *.png maps for foreground/background segmentation - all moving instances as foreground
└── instance_motion_segmentation_png/
    └── instance motion segmentation *.png maps
```

## DAVIS-Moving - COCO format
COCO format files for Raptor's evaluation on the Davis dataset[^ponttuset2017].

### Download
```
cd YOUR_davis_directory
gdown https://drive.google.com/uc?id=1gwet3yr7PVPVGcPUxKKounUuCRA39GA3 -O ./DAVISMoving_coco.zip
unzip DAVISMoving_coco.zip
```

### Data Description
```
├── readme.txt
│   └── file with link to this repo
└── motion_segmentation_coco_format/
    └── directory contating all evaluation COCO-format files
```

## YTVOS-Moving - COCO format
COCO format files for Raptor's evaluation on the YTVOS dataset[^xu2018].

### Download
```
cd YOUR_ytvos_directory
gdown https://drive.google.com/uc?id=162PgV286GShu0m7tXrcERzP3DPajtkio -O ./YTVOSMoving_coco.zip
unzip YTVOSMoving_coco.zip
```

### Data Description
```
├── readme.txt
│   └── file with link to this repo
└── motion_segmentation_coco_format/
    └── directory contating all evaluation COCO-format files
```


If you use this work please cite:
<pre>
@InProceedings{
    Neoral2021,
    author    = {Neoral, Michal and {\v{S}}ochman, Jan and Matas, Ji{\v{r}}{\'i}},
    title     = {Monocular Arbitrary Moving Object Discovery and Segmentation},
    booktitle = {The 32nd British Machine Vision Conference -- BMVC 2021},
    year      = {2021},
    }
</pre>

## License
Copyright (c) 2021 Toyota Motor Europe<br>
Patent Pending. All rights reserved.

Author: Michal Neoral, CMP FEE CTU Prague
Contact: neoramic@fel.cvut.cz

This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>

## References

[^menze2015]: Menze, M. & Geiger, A. Object Scene Flow for Autonomous Vehicles. CVPR, 2015.
[^mayer2016]: Mayer, N.; Ilg, E.; Hausser, P.; Fischer, P.; Cremers, D.; Dosovitskiy, A. & Brox, T. A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation. CVPR, 2016.
[^ponttuset2017]: Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017.
[^xu2018]: Xun Xu, Loong Fah Cheong, and Zhuwen Li. Motion segmentation by exploiting complementary geometric models. CVPR, 2018.