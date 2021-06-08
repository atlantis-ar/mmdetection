import torch.nn as nn

from mmdet.core import bbox2result

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class SingleStageSegDetector(BaseDetector):
    """Base class for single-stage instance segmenters.

    Single-stage instance segmenters directly and densely predict object masks
    on the output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_feat_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(SingleStageSegDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if mask_feat_head is not None:
            self.mask_feat_head = build_head(mask_feat_head)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(SingleStageSegDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
        else:
            self.neck.init_weights()
        if self.with_mask_feat_head:
            if isinstance(self.mask_feat_head, nn.Sequential):
                for m in self.mask_feat_head:
                    m.init_weights()
            else:
                self.mask_feat_head.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)

        outs = self.bbox_head(x)

        # loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas,
        # self.train_cfg)

        if self.with_mask_feat_head:
            mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                      start_level:self.mask_feat_head.end_level + 1])
            loss_inputs = outs + (mask_feat_pred, gt_bboxes, gt_labels,
                                  gt_masks, img_metas, self.train_cfg)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas,
                                  self.train_cfg)

        losses1 = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        # WES TODO: or compute losses like this:
        # losses2 = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore,
        #                                       gt_masks)

        return losses1  # 2

    def simple_test(self, img, img_meta, rescale=False):
        """Test function without test time augmentation."""
        x = self.extract_feat(img)
        outs = self.bbox_head(x, eval=True)

        if self.with_mask_feat_head:
            mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                      start_level:self.mask_feat_head.end_level + 1])
            seg_inputs = outs + (mask_feat_pred, img_meta, self.test_cfg,
                                 rescale)
        else:
            seg_inputs = outs + (img_meta, self.test_cfg, rescale)

        # # Note WES: use this implementation for test_ins_vis.py
        #seg_result = self.bbox_head.get_seg(*seg_inputs)
        # code from solov2 github WXinlong
        #return seg_result

        # # Note WES: use this implementation for inference, e.g. wes_python_demo.py
        # seg_inputs = outs + (img_meta, self.test_cfg, rescale)
        # code is also like this in mmdetection (v2.3) SOLO branch
        bbox_results, segm_results = self.bbox_head.get_seg(*seg_inputs)
        return bbox_results[0], segm_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
