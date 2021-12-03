_base_ = [
    '../_base_/datasets/scannet_instance.py'  # ,
    #    ' ../_base_/datasets/coco_instance_semantic.py',
    #    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='SOLOv2',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # C2, C3, C4, C5
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    bbox_head=dict(
        type='SOLOv2Head',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        seg_feat_channels=512,  # 256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        ins_out_channels=256,
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    mask_feat_head=dict(
        type='MaskFeatHead',
        in_channels=256,
        out_channels=128,
        start_level=0,
        end_level=3,
        num_classes=256,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),

    # cate_down_pos=0,
    # with_deform=False,
    # loss_ins=dict(
    #     type='DiceLoss',
    #     use_sigmoid=True,
    #     loss_weight=3.0),
    # loss_cate=dict(
    #     type='FocalLoss',
    #     use_sigmoid=True,
    #     gamma=2.0,
    #     alpha=0.25,
    #     loss_weight=1.0),
)
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=500,
    score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.05,
    kernel='gaussian',  # gaussian/linear
    sigma=2.0,
    max_per_img=100)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[27, 33])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 3  # 36
device_ids = range(1)  # 8)
dist_params = dict(backend='nccl')
log_level = 'DEBUG'
work_dir = './work_dirs/solov2_r101_3x_scannet_short'
load_from = None
resume_from = None
workflow = [('train', 1)]
