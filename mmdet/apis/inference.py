import warnings

import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the
            model will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None

    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)

        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    # data = collate([data], samples_per_gpu=1)
    # if next(model.parameters()).is_cuda:
    #     # scatter to specified GPU
    #     data = scatter(data, [device])[0]
    # else:
    #     # Use torchvision ops for CPU mode instead
    #     for m in model.modules():
    #         if isinstance(m, (RoIPool, RoIAlign)):
    #             if not m.aligned:
    #                 # aligned=False is not implemented on CPU
    #                 # set use_torchvision on-the-fly
    #                 m.use_torchvision = True
    #     warnings.warn('We set use_torchvision=True in CPU mode.')
    #     # just get the actual data from DataContainer
    #     data['img_meta'] = data['img_metas'][0].data
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    return result


async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       fig_size=(15, 10),
                       out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        out_file (str): If specified, the visualization result will
            be written to the out file.
    """
    if hasattr(model, 'module'):
        model = model.module

    show = False
    img = model.show_result(
        img, result, score_thr=score_thr, show=show, out_file=out_file)

    if show:
        plt.figure(figsize=fig_size)
        plt.imshow(mmcv.bgr2rgb(img))
        plt.show()

    if out_file:
        print('saving result to ' + out_file)
        # filename = "result.png"
        plt.imsave(out_file, mmcv.bgr2rgb(img))


# def close_contour(contour):
#     if not np.array_equal(contour[0],contour[-1]):
#         contour = np.vstack((contour, contour[0]))
#     return contour

# def binary_mask_to_polygon(binary_mask, tolerance=0):
#    polygons = []
#    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant',
#        constant_values=0)
#     contours = measure.find_contours(padded_binary_mask, 0.5)
#     contours = np.subtract(contours, 1)
#     for contour in contours:
#         contour = close_contour(contour)
#         contour = measure.approximate_polygon(contour, tolerance)
#         if len(contour) < 3:
#             continue
#         contour = np.flip(contour, axis=1)
#         segmentation = contour.ravel().tolist()
#         segmentation = [0 if i < 0 else i for i in segmentation]
#         polygons.append(segmentation)
#
#     return polygons

# def load_label_mapping(mapping_file_path):
#     """
#     Load a JSON mapping { class ID -> friendly class name }.
#     Used in BaseHandler.
#     """
#     if not os.path.isfile(mapping_file_path):
#         #logger.warning('Missing the index_to_name.json file. Inference
#             output will not include class name.')
#         return None
#
#     with open(mapping_file_path) as f:
#         mapping = json.load(f)
#     if not isinstance(mapping, dict):
#         raise Exception('index_to_name mapping should be in
#             "class":"label" json format')
#
#     # Older examples had a different syntax than others. This code
#         accommodates those.
#     if 'object_type_names' in mapping and
#         isinstance(mapping['object_type_names'], list):
#         mapping = {str(k): v for k, v in
#         enumerate(mapping['object_type_names'])}
#         return mapping
#
#     for key, value in mapping.items():
#         new_value = value
#         if isinstance(new_value, list):
#             new_value = value[-1]
#         if not isinstance(new_value, str):
#             raise Exception('labels in index_to_name must be either str or
#                  [str]')
#         mapping[key] = new_value
#     return mapping

# from inference copied
# # TODO: merge this method with the one in BaseDetector
# def show_result(img,
#                 result,
#                 class_names,
#                 score_thr=0.3,
#                 wait_time=0,
#                 show=True,
#                 out_file=None):
#     """Visualize the detection results on the image.
#
#     Args:
#         img (str or np.ndarray): Image filename or loaded image.
#         result (tuple[list] or list): The detection result, can be either
#             (bbox, segm) or just bbox.
#         class_names (list[str] or tuple[str]): A list of class names.
#         score_thr (float): The threshold to visualize the bboxes and masks.
#         wait_time (int): Value of waitKey param.
#         show (bool, optional): Whether to show the image with opencv or not.
#         out_file (str, optional): If specified, the visualization result
#             will be written to the out file instead of shown in a window.
#
#     Returns:
#         np.ndarray or None: If neither `show` nor `out_file` is specified,
#         the visualized image is returned, otherwise None is returned.
#     """
#
#     mapping_file_path = os.path.join(
#     "/home/digital/projects/torchServe/serve/examples/
#         image_segmenter_json/solov2/",
#         "index_to_name.json")
# #mapping = load_label_mapping(mapping_file_path)
# assert isinstance(class_names, (tuple, list))
# img = mmcv.imread(img)
#
# if isinstance(result, tuple):
#     bbox_result, segm_result = result
#     # 2 lines from torchserve postprocess
#     if isinstance(segm_result, tuple):
#         segm_result = segm_result[0]  # ms rcnn
# else:
#     bbox_result, segm_result = result, None
#
# bboxes_t_p_ = np.vstack(bbox_result)
# labels_t_p_ = [
#     np.full(bbox.shape[0], i, dtype=np.int32)
#     for i, bbox in enumerate(bbox_result)
# ]
# labels_t_p_ = np.concatenate(labels_t_p_)
#
# objects_list = []
#
# if segm_result is not None and len(labels_t_p_) > 0:  # non empty
#     segms_t_p_ = mmcv.concat_list(segm_result)
#     inds_t_p_ = np.where(bboxes_t_p_[:, -1] > score_thr)[0]
#     bboxes_conf_t_p_ = bboxes_t_p_[:, -1]
#     # inds = np.where(scores[:] > score_thr)[0]
#     inds = np.where(bboxes_t_p_[:, -1] > score_thr)[0]
#     np.random.seed(42)
#     color_masks = [
#         np.random.randint(0, 256, (1, 3), dtype=np.uint8)
#         for _ in range(max(labels_t_p_) + 1)
#     ]
#     # object_counter = 0
#     # tolerance = 2
#     for i in inds:
#         mask = segms_t_p_[i]
#         # polygon = binary_mask_to_polygon(mask, tolerance)
#         ii = np.nonzero(mask == True)
#         # json_entry_id = mapping[str(labels_t_p_[i])]
#         # json_entry_box = [int(min(ii[1])), int(min(ii[0])),
#         int(max(ii[1])),
#         int(max(ii[0]))]
#         i = int(i)
#         color_mask = color_masks[labels_t_p_[i]]
#         # mask = maskUtils.decode(segms[i]).astype(np.bool)
#         # mask = segm_result[i]
#         img[mask] = img[mask] * 0.3 + color_mask * 0.7
#
# if not (show or out_file):
#     return img
