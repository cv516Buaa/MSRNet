_base_ = './htc_r50_fpn_1x_coco.py'



dataset_type = 'CocoDataset'
classes = ('cell',)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=0,
    train=dict(
        img_prefix='dsb2018/train/',
        seg_prefix='dsb2018/stuffthingmaps/',
        density_prefix='dsb2018/train/densitymaps/',
        center_prefix='dsb2018/train/centermaps/',
        classes=classes,
        ann_file='dsb2018/train/annotation_coco.json'),
    val=dict(
        img_prefix='dsb2018/val/',
        density_prefix='dsb2018/train/densitymaps/',
        classes=classes,
        ann_file='dsb2018/val/annotation_coco.json'),
    test=dict(
        img_prefix='dsb2018/test/',
        density_prefix='dsb2018/train/densitymaps/',
        classes=classes,
        ann_file='dsb2018/test/annotation_coco.json'))
runner = dict(
    max_epochs=60)


# model settings
model = dict(
    type='SCNet',
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    roi_head=dict(
        _delete_=True,
        type='SCNetRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='SCNetBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='SCNetBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='SCNetBBoxHead',
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='SCNetMaskHead',
            num_convs=12,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            conv_to_res=True,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='SCNetSemanticHead',
            num_ins=5,
            fusion_level=0,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=256,
            loss_seg=dict(
                type='CrossEntropyLoss', loss_weight=0.2),
            conv_to_res=True),
        glbctx_head=dict(
            type='GlobalContextHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_weight=3.0,
            conv_to_res=True),
        feat_relay_head=dict(
            type='FeatureRelayHead',
            in_channels=1024,
            out_conv_channels=256,
            roi_feat_size=7,
            scale_factor=2),
        density_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        density_head=dict(
            type='SCNetDensityHead',
            num_ins=5,
            fusion_level=0,
            num_convs=5,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_seg=dict(
                type='MSELoss', loss_weight=0.003),
            conv_to_res=True),
        center_head=dict(
            type='SCNetDensityHead',
            num_ins=5,
            fusion_level=0,
            num_convs=5,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_center=dict(
                type='MSELoss', loss_weight=0.035),
            loss_hw=dict(
                type='SmoothL1Loss', loss_weight=40),
            conv_to_res=True)))


optimizer = dict(_delete_=True, type='AdamW', lr=0.0004, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))

# uncomment below code to enable test time augmentations
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=[(600, 900), (800, 1200), (1000, 1500), (1200, 1800),
#                    (1400, 2100)],
#         flip=True,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip', flip_ratio=0.5),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
