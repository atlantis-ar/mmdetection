import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
#from pycocotools.coco import COCO
#from pycocotools.cocoeval import COCOeval
#from terminaltables import AsciiTable

from .builder import DATASETS
#from .custom import CustomDataset
from .coco import CocoDataset

@DATASETS.register_module()
class ScannetDataset(CocoDataset):
    CLASSES = (
        # # Coco
        # 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        #       'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
        #      'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
        #       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        #       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        #       'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
        #       'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
        #       'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        #       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        #       'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        #      'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
        #      'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        #      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        #      'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
        # scannet
        'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
        'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves',
        'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling',
        'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain',
        'box', 'whiteboard', 'person', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub',
        'bag', 'otherstructure', 'otherfurniture', 'otherprop')
    # overlap for evaluation:
    #'bed'(coco index <=> scannet index: x <=> 4
    #'chair' (coco index <=> scannet index: x <=> 5
    #'sofa'<=> couch???
    #'table'/desk <=> dining table
    #'books'<=> book,
    #'refridgerator' <=> refrigerator
    #'television'<=> tv,
    #'person',
    # 'toilet',
    # sink',
    #'bag' <=> handbag ,

    # def load_annotations(self, ann_file):
    #     """Load annotation from COCO style annotation file.
    #
    #     Args:
    #         ann_file (str): Path of annotation file.
    #
    #     Returns:
    #         list[dict]: Annotation info from COCO api.
    #     """
    #
    #     self.coco = COCO(ann_file)
    #     self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
    #     self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
    #     self.img_ids = self.coco.getImgIds()
    #     data_infos = []
    #     for i in self.img_ids:
    #         info = self.coco.loadImgs([i])[0]
    #         info['filename'] = info['file_name']
    #         data_infos.append(info)
    #     return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            all_iscrowd = all([_['iscrowd'] for _ in ann_info])
            if self.filter_empty_gt and (self.img_ids[i] not in ids_with_ann
                                         or all_iscrowd):
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)


def get_cat_ids(self, idx):
    """Get COCO category ids by index.

    Args:
        idx (int): Index of data.

    Returns:
        list[int]: All categories in the image of specified index.
    """

    img_id = self.data_infos[idx]['id']
    ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
    ann_info = self.coco.load_anns(ann_ids)
    return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    # def get_subset_by_classes(self):
    #     """Get img ids that contain any category in class_ids.
    #
    #     Different from the coco.getImgIds(), this function returns the id if
    #     the img contains one of the categories rather than all.
    #
    #     Args:
    #         class_ids (list[int]): list of category ids
    #
    #     Return:
    #         ids (list[int]): integer list of img ids
    #     """
    #
    #     ids = set()
    #     for i, class_id in enumerate(self.cat_ids):
    #         ids |= set(self.coco.cat_img_map[class_id])
    #     self.img_ids = list(ids)
    #
    #     data_infos = []
    #     for i in self.img_ids:
    #         info = self.coco.load_imgs([i])[0]
    #         info['filename'] = info['file_name']
    #         data_infos.append(info)
    #     return data_infos


    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

        # def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        #     gt_bboxes = []
        #     for i in range(len(self.img_ids)):
        #         ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
        #         ann_info = self.coco.load_anns(ann_ids)
        #         if len(ann_info) == 0:
        #             gt_bboxes.append(np.zeros((0, 4)))
        #             continue
        #         bboxes = []
        #         for ann in ann_info:
        #             if ann.get('ignore', False) or ann['iscrowd']:
        #                 continue
        #             x1, y1, w, h = ann['bbox']
        #             bboxes.append([x1, y1, x1 + w, y1 + h])
        #         bboxes = np.array(bboxes, dtype=np.float32)
        #         if bboxes.shape[0] == 0:
        #             bboxes = np.zeros((0, 4))
        #         gt_bboxes.append(bboxes)
        #
        #     recalls = eval_recalls(
        #         gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        #     ar = recalls.mean(axis=1)
        #     return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir


    # def results2txt(self, results, outfile_prefix):
    #     """Dump the detection results to a txt file.
    #
    #     Args:
    #         results (list[list | tuple]): Testing results of the
    #             dataset.
    #         outfile_prefix (str): The filename prefix of the json files.
    #             If the prefix is "somepath/xxx",
    #             the txt files will be named "somepath/xxx.txt".
    #
    #     Returns:
    #         list[str]: Result txt files which contains corresponding \
    #             instance segmentation images.
    #     """
    #     try:
    #         import cityscapesscripts.helpers.labels as CSLabels
    #     except ImportError:
    #         raise ImportError('Please run "pip install citscapesscripts" to '
    #                           'install cityscapesscripts first.')
    #     result_files = []
    #     os.makedirs(outfile_prefix, exist_ok=True)
    #     prog_bar = mmcv.ProgressBar(len(self))
    #     for idx in range(len(self)):
    #         result = results[idx]
    #         filename = self.data_infos[idx]['filename']
    #         basename = osp.splitext(osp.basename(filename))[0]
    #         pred_txt = osp.join(outfile_prefix, basename + '_pred.txt')
    #
    #         bbox_result, segm_result = result
    #         bboxes = np.vstack(bbox_result)
    #         # segm results
    #         if isinstance(segm_result, tuple):
    #             # Some detectors use different scores for bbox and mask,
    #             # like Mask Scoring R-CNN. Score of segm will be used instead
    #             # of bbox score.
    #             segms = mmcv.concat_list(segm_result[0])
    #             mask_score = segm_result[1]
    #         else:
    #             # use bbox score for mask score
    #             segms = mmcv.concat_list(segm_result)
    #             mask_score = [bbox[-1] for bbox in bboxes]
    #         labels = [
    #             np.full(bbox.shape[0], i, dtype=np.int32)
    #             for i, bbox in enumerate(bbox_result)
    #         ]
    #         labels = np.concatenate(labels)
    #
    #         assert len(bboxes) == len(segms) == len(labels)
    #         num_instances = len(bboxes)
    #         prog_bar.update()
    #         with open(pred_txt, 'w') as fout:
    #             for i in range(num_instances):
    #                 pred_class = labels[i]
    #                 classes = self.CLASSES[pred_class]
    #                 class_id = CSLabels.name2label[classes].id
    #                 score = mask_score[i]
    #                 mask = maskUtils.decode(segms[i]).astype(np.uint8)
    #                 png_filename = osp.join(outfile_prefix,
    #                                         basename + f'_{i}_{classes}.png')
    #                 mmcv.imwrite(mask, png_filename)
    #                 fout.write(f'{osp.basename(png_filename)} {class_id} '
    #                            f'{score}\n')
    #         result_files.append(pred_txt)
    #
    #     return result_files
    #
    # def format_results(self, results, txtfile_prefix=None):
    #     """Format the results to txt (standard format for Cityscapes
    #     evaluation).
    #
    #     Args:
    #         results (list): Testing results of the dataset.
    #         txtfile_prefix (str | None): The prefix of txt files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.
    #
    #     Returns:
    #         tuple: (result_files, tmp_dir), result_files is a dict containing \
    #             the json filepaths, tmp_dir is the temporal directory created \
    #             for saving txt/png files when txtfile_prefix is not specified.
    #     """
    #     assert isinstance(results, list), 'results must be a list'
    #     assert len(results) == len(self), (
    #         'The length of results is not equal to the dataset len: {} != {}'.
    #         format(len(results), len(self)))
    #
    #     assert isinstance(results, list), 'results must be a list'
    #     assert len(results) == len(self), (
    #         'The length of results is not equal to the dataset len: {} != {}'.
    #         format(len(results), len(self)))
    #
    #     if txtfile_prefix is None:
    #         tmp_dir = tempfile.TemporaryDirectory()
    #         txtfile_prefix = osp.join(tmp_dir.name, 'results')
    #     else:
    #         tmp_dir = None
    #     result_files = self.results2txt(results, txtfile_prefix)
    #
    #     return result_files, tmp_dir
    #
    # def evaluate(self,
    #              results,
    #              metric='bbox',
    #              logger=None,
    #              outfile_prefix=None,
    #              classwise=False,
    #              proposal_nums=(100, 300, 1000),
    #              iou_thrs=np.arange(0.5, 0.96, 0.05)):
    #     """Evaluation in Cityscapes/COCO protocol.
    #
    #     Args:
    #         results (list[list | tuple]): Testing results of the dataset.
    #         metric (str | list[str]): Metrics to be evaluated. Options are
    #             'bbox', 'segm', 'proposal', 'proposal_fast'.
    #         logger (logging.Logger | str | None): Logger used for printing
    #             related information during evaluation. Default: None.
    #         outfile_prefix (str | None): The prefix of output file. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If results are evaluated with COCO protocol, it would be the
    #             prefix of output json file. For example, the metric is 'bbox'
    #             and 'segm', then json files would be "a/b/prefix.bbox.json" and
    #             "a/b/prefix.segm.json".
    #             If results are evaluated with cityscapes protocol, it would be
    #             the prefix of output txt/png files. The output files would be
    #             png images under folder "a/b/prefix/xxx/" and the file name of
    #             images would be written into a txt file
    #             "a/b/prefix/xxx_pred.txt", where "xxx" is the video name of
    #             cityscapes. If not specified, a temp file will be created.
    #             Default: None.
    #         classwise (bool): Whether to evaluating the AP for each class.
    #         proposal_nums (Sequence[int]): Proposal number used for evaluating
    #             recalls, such as recall@100, recall@1000.
    #             Default: (100, 300, 1000).
    #         iou_thrs (Sequence[float]): IoU threshold used for evaluating
    #             recalls. If set to a list, the average recall of all IoUs will
    #             also be computed. Default: 0.5.
    #
    #     Returns:
    #         dict[str, float]: COCO style evaluation metric or cityscapes mAP \
    #             and AP@50.
    #     """
    #     eval_results = dict()
    #
    #     metrics = metric.copy() if isinstance(metric, list) else [metric]
    #
    #     if 'cityscapes' in metrics:
    #         eval_results.update(
    #             self._evaluate_cityscapes(results, outfile_prefix, logger))
    #         metrics.remove('cityscapes')
    #
    #     # left metrics are all coco metric
    #     if len(metrics) > 0:
    #         # create CocoDataset with CityscapesDataset annotation
    #         self_coco = CocoDataset(self.ann_file, self.pipeline.transforms,
    #                                 None, self.data_root, self.img_prefix,
    #                                 self.seg_prefix, self.proposal_file,
    #                                 self.test_mode, self.filter_empty_gt)
    #         # TODO: remove this in the future
    #         # reload annotations of correct class
    #         self_coco.CLASSES = self.CLASSES
    #         self_coco.data_infos = self_coco.load_annotations(self.ann_file)
    #         eval_results.update(
    #             self_coco.evaluate(results, metrics, logger, outfile_prefix,
    #                                classwise, proposal_nums, iou_thrs))
    #
    #     return eval_results
    #
    # def _evaluate_cityscapes(self, results, txtfile_prefix, logger):
    #     """Evaluation in Cityscapes protocol.
    #
    #     Args:
    #         results (list): Testing results of the dataset.
    #         txtfile_prefix (str | None): The prefix of output txt file
    #         logger (logging.Logger | str | None): Logger used for printing
    #             related information during evaluation. Default: None.
    #
    #     Returns:
    #         dict[str: float]: Cityscapes evaluation results, contains 'mAP' \
    #             and 'AP@50'.
    #     """
    #
    #     try:
    #         import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as CSEval  # noqa
    #     except ImportError:
    #         raise ImportError('Please run "pip install citscapesscripts" to '
    #                           'install cityscapesscripts first.')
    #     msg = 'Evaluating in Cityscapes style'
    #     if logger is None:
    #         msg = '\n' + msg
    #     print_log(msg, logger=logger)
    #
    #     result_files, tmp_dir = self.format_results(results, txtfile_prefix)
    #
    #     if tmp_dir is None:
    #         result_dir = osp.join(txtfile_prefix, 'results')
    #     else:
    #         result_dir = osp.join(tmp_dir.name, 'results')
    #
    #     eval_results = {}
    #     print_log(f'Evaluating results under {result_dir} ...', logger=logger)
    #
    #     # set global states in cityscapes evaluation API
    #     CSEval.args.cityscapesPath = os.path.join(self.img_prefix, '../..')
    #     CSEval.args.predictionPath = os.path.abspath(result_dir)
    #     CSEval.args.predictionWalk = None
    #     CSEval.args.JSONOutput = False
    #     CSEval.args.colorized = False
    #     CSEval.args.gtInstancesFile = os.path.join(result_dir,
    #                                                'gtInstances.json')
    #     CSEval.args.groundTruthSearch = os.path.join(
    #         self.img_prefix.replace('leftImg8bit', 'gtFine'),
    #         '*/*_gtFine_instanceIds.png')
    #
    #     groundTruthImgList = glob.glob(CSEval.args.groundTruthSearch)
    #     assert len(groundTruthImgList), 'Cannot find ground truth images' \
    #         f' in {CSEval.args.groundTruthSearch}.'
    #     predictionImgList = []
    #     for gt in groundTruthImgList:
    #         predictionImgList.append(CSEval.getPrediction(gt, CSEval.args))
    #     CSEval_results = CSEval.evaluateImgLists(predictionImgList,
    #                                              groundTruthImgList,
    #                                              CSEval.args)['averages']
    #
    #     eval_results['mAP'] = CSEval_results['allAp']
    #     eval_results['AP@50'] = CSEval_results['allAp50%']
    #     if tmp_dir is not None:
    #         tmp_dir.cleanup()
    #     return eval_results
    #
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls/mAPs. If set to a list, the average of all IoUs will
                also be computed. Default: np.arange(0.5, 0.96, 0.05).

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.iouThrs = iou_thrs
            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]
                for i, item in enumerate(metric_items):
                    val = float(f'{cocoEval.stats[i + 6]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]
                for i in range(len(metric_items)):
                    key = f'{metric}_{metric_items[i]}'
                    val = float(f'{cocoEval.stats[i]:.3f}')
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
