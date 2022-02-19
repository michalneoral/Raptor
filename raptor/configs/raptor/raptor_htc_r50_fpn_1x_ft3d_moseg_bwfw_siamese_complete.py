_base_ = './raptor_htc_without_semantic_r50_fpn_1x_ft3d_moseg_bwfw_siamese_complete.py'
model = dict(
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=512,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            num_convs=4,
            in_channels=512,
            conv_out_channels=512,
            num_classes=2,
            ignore_label=255,
            loss_weight=0.2)))
# data_root = '/datagrid/public_datasets/SceneFlowMayer/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadTwoImagesFromFiles'),
#     dict(
#         type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True, change_semantic_labels=[(255,1)]),
#     dict(type='Resize', img_scale=(960, 540), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.0), # NO FLIP! default was 0.5
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='SegRescale', scale_factor=1 / 8),
#     dict(type='DefaultFormatBundle'),
#     dict(
#         type='Collect',
#         keys=['img', 'img2', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'calib_position', 'calib_position_2', 'calib_K', 'calib_K_2', 'calib_baseline']),
#         #keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]
# test_pipeline = [
#     dict(type='LoadTwoImagesFromFiles'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(960, 540),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip', flip_ratio=0.0), # NO FLIP! default was 0.5
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img', 'img2']),
#             dict(type='Collect', keys=['img', 'img2', 'calib_K', 'calib_K_2', 'calib_baseline']),
#         ])
# ]
# data = dict(
#     train=dict(
#         seg_prefix=data_root + 'binary_motion_segmentation_png/',  # + 'stuffthingmaps/train2017/',
#         pipeline=train_pipeline),
#     # train=dict(
#     #     type='ConcatDataset',
#     #     separate_eval=False,
#     #     datasets=[
#     #         dict(
#     #             seg_prefix=data_root + 'binary_motion_segmentation_png/',  # + 'stuffthingmaps/train2017/',
#     #             pipeline=train_pipeline),
#     #         dict(
#     #             seg_prefix=data_root + 'binary_motion_segmentation_png/',  # + 'stuffthingmaps/train2017/',
#     #             pipeline=train_pipeline),
#     #         dict(
#     #             seg_prefix=data_root + 'binary_motion_segmentation_png/',  # + 'stuffthingmaps/train2017/',
#     #             pipeline=train_pipeline),
#     #         dict(
#     #             seg_prefix=data_root + 'binary_motion_segmentation_png/',  # + 'stuffthingmaps/train2017/',
#     #             pipeline=train_pipeline),
#     #     ]
#     # ),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
