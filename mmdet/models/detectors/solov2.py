from ..builder import DETECTORS
from .single_stage_seg import SingleStageSegDetector


@DETECTORS.register_module()
class SOLOv2(SingleStageSegDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_feat_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SOLOv2, self).__init__(backbone, neck, bbox_head, mask_feat_head,
                                     train_cfg, test_cfg, pretrained)
