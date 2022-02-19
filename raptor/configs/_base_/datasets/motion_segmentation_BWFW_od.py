dataset_type_eval_kitti = 'KittiCocoBasedMotionSegmentationDataset'
dataset_type_eval_kitti_new = 'KittiNewCocoBasedMotionSegmentationDataset'

dataset_type_eval_moonka = 'MoonkaCocoBasedMotionSegmentationDataset'
dataset_type_eval_driving = 'DrivingCocoBasedMotionSegmentationDataset'
dataset_type_eval_ft3d = 'FlyingThingsCocoBasedMotionSegmentationDataset'
dataset_type_eval_davis = 'DavisCocoBasedMotionSegmentationDataset'
dataset_type_eval_ytvos = 'YtvosCocoBasedMotionSegmentationDataset'

dataset_type = dataset_type_eval_ft3d

data_root = '/datagrid/public_datasets/SceneFlowMayer/'
data_root_kitti = '/datagrid/public_datasets/KITTI/multiview/training/'
data_root_kitti_new = '/datagrid/public_datasets/KITTI/multiview/training/'
data_root_davis = '/datagrid/public_datasets/DAVIS/'
data_root_ytvos = '/datagrid/public_datasets/YouTubeVOS_2018/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadTwoImagesFromFiles'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True, change_semantic_labels=[(255,1)]),
    dict(type='Resize', img_scale=(960, 540), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0), # NO FLIP! default was 0.5
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle_Custom'),
    dict(
        type='Collect_Custom',
        keys=['img', 'img2', 'img0', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'calib_position', 'calib_position_2', 'calib_position_0', 'calib_K', 'calib_K_2', 'calib_K_0', 'calib_baseline']),
        #keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadTwoImagesFromFiles'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 540),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0), # NO FLIP! default was 0.5
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor_Custom', keys=['img', 'img2', 'img0', 'calib_K', 'calib_K_2', 'calib_K_0', 'calib_baseline']),
            # dict(type='DefaultFormatBundle_Custom'),
            dict(type='Collect_Custom', keys=['img', 'img2', 'img0', 'calib_K', 'calib_K_2', 'calib_K_0', 'calib_baseline']),
        ])
]
test_pipeline_kitti = [
    dict(type='LoadTwoImagesFromFiles'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1242, 376),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0), # NO FLIP! default was 0.5
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor_Custom', keys=['img', 'img2', 'img0', 'calib_K', 'calib_K_2', 'calib_K_0', 'calib_baseline']),
            # dict(type='DefaultFormatBundle_Custom'),
            dict(type='Collect_Custom', keys=['img', 'img2', 'img0', 'calib_K', 'calib_K_2', 'calib_K_0', 'calib_baseline']),
        ])
]
test_pipeline_kitti_new = [
    dict(type='LoadTwoImagesFromFiles'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1242, 376),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0), # NO FLIP! default was 0.5
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor_Custom', keys=['img', 'img2', 'img0', 'calib_K', 'calib_K_2', 'calib_K_0', 'calib_baseline']),
            # dict(type='DefaultFormatBundle_Custom'),
            dict(type='Collect_Custom', keys=['img', 'img2', 'img0', 'calib_K', 'calib_K_2', 'calib_K_0', 'calib_baseline']),
        ])
]
test_pipeline_davis = [
    dict(type='LoadTwoImagesFromFiles'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(854, 480),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0), # NO FLIP! default was 0.5
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor_Custom', keys=['img', 'img2', 'img0', 'calib_K', 'calib_K_2', 'calib_K_0', 'calib_baseline']),
            # dict(type='DefaultFormatBundle_Custom'),
            dict(type='Collect_Custom', keys=['img', 'img2', 'img0', 'calib_K', 'calib_K_2', 'calib_K_0', 'calib_baseline']),
        ])
]
test_pipeline_ytvos = [
    dict(type='LoadTwoImagesFromFiles'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0), # NO FLIP! default was 0.5
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor_Custom', keys=['img', 'img2', 'img0', 'calib_K', 'calib_K_2', 'calib_K_0', 'calib_baseline']),
            # dict(type='DefaultFormatBundle_Custom'),
            dict(type='Collect_Custom', keys=['img', 'img2', 'img0', 'calib_K', 'calib_K_2', 'calib_K_0', 'calib_baseline']),
        ])
]
evaluation = dict(metric=['bbox', 'segm'], kitti_metric=['obj', 'bg'], nproc=8)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        separate_eval=False,
        datasets=[
            dict(
                type=dataset_type,
                ann_file=data_root +
                'motion_segmentation_coco_format/moving_objects_driving_final_bwfw_od_train.json',
                img_prefix=data_root,
                seg_prefix=data_root + 'binary_motion_segmentation_png/',
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root +
                'motion_segmentation_coco_format/moving_objects_driving_clean_bwfw_od_train.json',
                img_prefix=data_root,
                seg_prefix=data_root + 'binary_motion_segmentation_png/',
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root +
                'motion_segmentation_coco_format/moving_objects_ft3d_clean_bwfw_od_train.json',
                img_prefix=data_root,
                seg_prefix=data_root + 'binary_motion_segmentation_png/',
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root +
                'motion_segmentation_coco_format/moving_objects_moonka_clean_bwfw_od_train.json',
                img_prefix=data_root,
                seg_prefix=data_root + 'binary_motion_segmentation_png/',
                pipeline=train_pipeline),
        ]
    ),
    val=dict(
        type='ConcatDataset',
        separate_eval=True,
        datasets=[
            dict(
                type=dataset_type_eval_kitti,
                ann_file=data_root_kitti +
                         'motion_segmentation_coco_format/moving_objects_kitti_original_bwfw_all.json',
                img_prefix=data_root_kitti,
                pipeline=test_pipeline_kitti),
            dict(
                type=dataset_type_eval_davis,
                ann_file=data_root_davis +
                         'motion_segmentation_coco_format/moving_objects_davis_small_bwfw_val.json',
                img_prefix=data_root_davis + 'DAVIS17/',
                pipeline=test_pipeline_davis),
            dict(
                type=dataset_type_eval_driving,
                ann_file=data_root +
                         'motion_segmentation_coco_format/moving_objects_driving_final_small_bwfw_val.json',
                img_prefix=data_root,
                pipeline=test_pipeline),
            dict(
                type=dataset_type_eval_ft3d,
                ann_file=data_root +
                         'motion_segmentation_coco_format/moving_objects_ft3d_clean_small_bwfw_val.json',
                img_prefix=data_root,
                pipeline=test_pipeline),
            dict(
                type=dataset_type_eval_moonka,
                ann_file=data_root +
                         'motion_segmentation_coco_format/moving_objects_moonka_clean_small_bwfw_val.json',
                img_prefix=data_root,
                pipeline=test_pipeline),
        ]
    ),
    test=dict(
        type='ConcatDataset',
        separate_eval=True,
        datasets=[
            dict(
                type=dataset_type_eval_kitti,
                ann_file=data_root_kitti +
                         'motion_segmentation_coco_format/moving_objects_kitti_original_eval.json',
                img_prefix=data_root_kitti,
                pipeline=test_pipeline_kitti),

            dict(
                type=dataset_type_eval_kitti_new,
                ann_file=data_root_kitti_new +
                         'motion_segmentation_coco_format/moving_objects_kitti_new_eval.json',
                img_prefix=data_root_kitti_new,
                pipeline=test_pipeline_kitti_new),
            # dict(
            #     type=dataset_type_eval_kitti_new,
            #     ann_file=data_root_kitti_new +
            #              'motion_segmentation_coco_format/to_track_kitti_new_eval.json',
            #     img_prefix=data_root_kitti_new,
            #     pipeline=test_pipeline_kitti_new),
            dict(
                type=dataset_type_eval_davis,
                ann_file=data_root_davis +
                         'motion_segmentation_coco_format/moving_objects_davis_original_eval.json',
                img_prefix=data_root_davis + 'DAVIS17/',
                pipeline=test_pipeline_davis),
            # dict(
            #     type=dataset_type_eval_davis,
            #     ann_file=data_root_davis +
            #              'motion_segmentation_coco_format/to_track_davis_original_eval.json',
            #     img_prefix=data_root_davis + 'DAVIS17/',
            #     pipeline=test_pipeline_davis),

            dict(
                type=dataset_type_eval_ytvos,
                ann_file=data_root_ytvos +
                         'motion_segmentation_coco_format/moving_objects_ytvos_original_eval.json',
                img_prefix=data_root_ytvos,
                pipeline=test_pipeline_ytvos),
            # dict(
            #     type=dataset_type_eval_ytvos,
            #     ann_file=data_root_ytvos +
            #              'motion_segmentation_coco_format/to_track_ytvos_original_eval.json',
            #     img_prefix=data_root_ytvos,
            #     pipeline=test_pipeline_ytvos),

            # dict(
            #     type=dataset_type_eval_kitti,
            #     ann_file=data_root_kitti +
            #              'motion_segmentation_coco_format/moving_objects_kitti_original_bwfw_all.json',
            #     img_prefix=data_root_kitti,
            #     pipeline=test_pipeline_kitti),
            # dict(
            #     type=dataset_type_eval_davis,
            #     ann_file=data_root_davis +
            #              'motion_segmentation_coco_format/moving_objects_davis_small_bwfw_all.json',
            #     img_prefix=data_root_davis + 'DAVIS17/',
            #     pipeline=test_pipeline_davis),
            # dict(
            #     type=dataset_type_eval_driving,
            #     ann_file=data_root +
            #              'motion_segmentation_coco_format/moving_objects_driving_final_small_bwfw_val.json',
            #     img_prefix=data_root,
            #     pipeline=test_pipeline),
            # dict(
            #     type=dataset_type_eval_ft3d,
            #     ann_file=data_root +
            #              'motion_segmentation_coco_format/moving_objects_ft3d_clean_small_bwfw_val.json',
            #     img_prefix=data_root,
            #     pipeline=test_pipeline),
            # dict(
            #     type=dataset_type_eval_moonka,
            #     ann_file=data_root +
            #              'motion_segmentation_coco_format/moving_objects_moonka_clean_small_bwfw_val.json',
            #     img_prefix=data_root,
            #     pipeline=test_pipeline),
        ]
    )
)
