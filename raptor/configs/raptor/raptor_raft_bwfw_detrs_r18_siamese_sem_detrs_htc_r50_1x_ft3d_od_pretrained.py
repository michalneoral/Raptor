_base_ = './raptor_htc_r50_fpn_1x_ft3d_moseg_bwfw_siamese_complete.py'

# import custom models
custom_imports = dict(
    imports=['raptor.mmdet_custom.models.detectors',
             'raptor.mmdet_custom.models.backbones',
             'raptor.mmdet_custom.datasets.pipelines',
             'raptor.mmdet_custom.datasets',
             #'raptor.mmdet_custom.models.motion_backbones'
             ],
    allow_failed_imports=False)

model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True,
        frozen_stages=4,
    ),
    motion_backbone=dict(
        type='MotionBackboneWithDetectoRS',
        backbone=dict(
            type='DetectoRS_ResNet_Custom',
            # type='DetectoRS_ResNet',
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            output_img=True,
            frozen_stages=4,
            in_channels=14,
            depth=18,
            # depth=50,
            pretrained='torchvision://resnet18',
        ),
        neck=dict(
            type='RFP',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            rfp_steps=2,
            aspp_out_channels=64,
            aspp_dilations=(1, 3, 6, 1),
            rfp_backbone=dict(
                rfp_inplanes=256,
                # type='DetectoRS_ResNet',
                type='DetectoRS_ResNet_Custom',
                # depth=50,
                depth=18,
                in_channels=14,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=4,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=True,
                conv_cfg=dict(type='ConvAWS'),
                sac=dict(type='SAC', use_deform=True),
                stage_with_sac=(False, True, True, True),
                pretrained='torchvision://resnet18',
                style='pytorch')),
        motion_cost_volume=dict(
            type='RAFTplus',
            siamese=True,
            siamese_outputs=128,
            max_disp=256,
            fac=1.,
            stage=2,
            samples_per_gpu=2,
            img_norm_cfg=dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            img_scale=(960, 540),
            #exp_unc='/datagrid/personal/neoral/repos/rigidmask_debug/weights/rigidmask-sf/weights.pth',
            exp_unc=True,
            num_stages=1,
            frozen_stages=2,
            dual_network=True,
            switched_camera_positions=False,
            use_opencv=True, # True: opencv - False: ngransac - for E,R,t estimation
            small=False,  # RAFT use small model
            iters=24,  # RAFT number of recurrent iterations
            mixed_precision=False,  # RAFT use mixed precision
            alternate_corr=False,  # RAFT use efficent correlation implementation
            #loadraft='/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/100000_raft-rob-100k.pth', # RAFT path of the pre-trained flow model
        ),
    ),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            #pretrained='torchvision://resnet50',
            style='pytorch')))
#load_from = '/home.stud/neoramic/repos/mmdetection_debug/checkpoints/combo_complete_raft_moseg_r18_detectors_r50.pth'
find_unused_parameters = True
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='raptor',
                entity='cmp',
                name='{{ fileBasenameNoExtension }}',
                notes='With RAFT BWFW motion segmentation part.',
            ))
    ])