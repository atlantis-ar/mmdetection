from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.utils import print_log

from mmdet.core import auto_fp16
from mmdet.utils import get_root_logger
from skimage import measure

class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_mask_feat_head(self):
        return hasattr(self, 'mask_feat_head') and \
               self.mask_feat_head is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.roi_head,
                       'shared_head') and self.roi_head.shared_head is not None

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.roi_head, 'bbox_head')
                 and self.roi_head.bbox_head is not None)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.roi_head, 'mask_head')
                 and self.roi_head.mask_head is not None)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if pretrained is not None:
            logger = get_root_logger()
            print_log(f'load model from: {pretrained}', logger=logger)

    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of image metas ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = imgs[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def binary_mask_to_polygon(self, mask, tolerance=0):
        polygons = []
        padded_binary_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        contours = np.subtract(contours, 1)
        for contour in contours:

            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack(contour, contour[0])

            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            segmentation = [0 if i < 0 else i for i in segmentation]
            polygons.append(segmentation)

        return polygons


    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        # r1, r2, r3 = bbox_result[0]
        # segm_result = r1.cpu()
        # labels = r2.cpu()
        # scores = r3.cpu()

        #bbox_result has to be a type {list: 80}
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]

        labels = np.concatenate(labels)

        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            #inds = np.where(scores[:] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            object_counter = 0
            polygons = []
            bboxes = []
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i]
                #mask = segm_result[i]
                img[mask] = img[mask] * 0.3 + color_mask * 0.7

                ii = np.nonzero(mask == True)
                #polygons = self.binary_mask_to_polygon(mask, tolerance)
                bbox = [int(min(ii[1])), int(min(ii[0])), int(max(ii[1])), int(max(ii[0]))]
                bboxes.append(bbox)
                object_counter = object_counter + 1
        # if out_file specified, do not show image in window
        # if out_file is not None:
        #    show = False
        # draw bounding boxes
        # mmcv.imshow_det_bboxes(
        #     img,
        #     bboxes,
        #     labels,
        #     class_names=self.CLASSES,
        #     score_thr=score_thr,
        #     bbox_color=bbox_color,
        #     text_color=text_color,
        #     thickness=thickness,
        #     font_scale=font_scale,
        #     win_name=win_name,
        #     show=show,
        #     wait_time=wait_time,
        #     out_file=out_file)

        # if not (show or out_file):
        return img


# def show_result(self, data, result, dataset=None, score_thr=0.3):
#     if isinstance(result, tuple):
#         bbox_result, segm_result = result
#     else:
#         bbox_result, segm_result = result, None
#
#     img_tensor = data['img'][0]
#     img_metas = data['img_meta'][0].data[0]
#     imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
#     assert len(imgs) == len(img_metas)
#
#     if dataset is None:
#         class_names = self.CLASSES
#     elif isinstance(dataset, str):
#         class_names = get_classes(dataset)
#     elif isinstance(dataset, (list, tuple)):
#         class_names = dataset
#     else:
#         raise TypeError(
#             'dataset must be a valid dataset name or a sequence'
#             ' of class names, not {}'.format(type(dataset)))
#
#     for img, img_meta in zip(imgs, img_metas):
#         h, w, _ = img_meta['img_shape']
#         img_show = img[:h, :w, :]
#
#         bboxes = np.vstack(bbox_result)
#         # draw segmentation masks
#         if segm_result is not None:
#             segms = mmcv.concat_list(segm_result)
#             inds = np.where(bboxes[:, -1] > score_thr)[0]
#             for i in inds:
#                 color_mask = np.random.randint(
#                     0, 256, (1, 3), dtype=np.uint8)
#                 mask = maskUtils.decode(segms[i]).astype(np.bool)
#                 img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
#         # draw bounding boxes
#         labels = [
#             np.full(bbox.shape[0], i, dtype=np.int32)
#             for i, bbox in enumerate(bbox_result)
#         ]
#         labels = np.concatenate(labels)
#         mmcv.imshow_det_bboxes(
#             img_show,
#             bboxes,
#             labels,
#             class_names=class_names,
#             score_thr=score_thr)
