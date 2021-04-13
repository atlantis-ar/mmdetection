import argparse
import os
import os.path as osp
from os import listdir
from os import path
import shutil
import tempfile

import cv2
import matplotlib.cm as cm
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
# for visualization
from mmcv.parallel import collate, scatter
from mmcv.runner import init_dist, get_dist_info, load_checkpoint
from mmcv.runner import load_checkpoint
from scipy import ndimage

from mmdet.core import coco_eval, results2json, wrap_fp16_model, tensor2imgs, get_classes
from mmdet.core import get_classes
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


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


def vis_seg(data, result, img_norm_cfg, data_id, colors, score_thr, save_dir):
    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0] #.data[0]
    # img_metas = data['img_metas'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_norm_cfg)
    assert len(imgs) == len(img_metas)
    class_names = get_classes('coco')
    #class_names = get_classes('scannet')

    for img, img_meta, cur_result in zip(imgs, img_metas, result):
        if cur_result is None:
            continue
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        
        seg_label = cur_result[0]
        seg_label = seg_label.cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1]
        cate_label = cate_label.cpu().numpy()
        #score = cur_result[2].cpu().numpy()
        score = cur_result[2].cpu().detach().numpy()

        vis_inds = score > score_thr
        seg_label = seg_label[vis_inds]
        num_mask = seg_label.shape[0]
        cate_label = cate_label[vis_inds]
        cate_score = score[vis_inds]

        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

        seg_show = img_show.copy()
        for idx in range(num_mask):
            idx = -(idx+1)
            cur_mask = seg_label[idx, :,:]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
               continue
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            cur_mask_bool = cur_mask.astype(np.bool)
            seg_show[cur_mask_bool] = img_show[cur_mask_bool] * 0.5 + color_mask * 0.5

            cur_cate = cate_label[idx]
            cur_score = cate_score[idx]

            #label_text = class_names[cur_cate]+" " +str(cur_cate)+" " +str(cur_score)
            label_text = class_names[cur_cate] + " (" + str(cur_score.round(2))+")"

            #label_text += '|{:.02f}'.format(cur_score)
            # center
            center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
            vis_pos = (max(int(center_x) - 10, 0), int(center_y))
            cv2.putText(seg_show, label_text, vis_pos,
                        cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255))  # green
        mmcv.imwrite(seg_show, '{}/{}.jpg'.format(save_dir, data_id))


def single_gpu_test(model, data_loader, args, cfg=None, verbose=True):
    model.eval()
    results = []
    dataset = data_loader.dataset

    class_num = 1000 # ins
    colors = [(np.random.random((1, 3)) * 255).tolist()[0] for i in range(class_num)]    

    prog_bar = mmcv.ProgressBar(len(dataset))

    # # origin code begin: for testing and visualizing img configured in the config file
    # for i, data in enumerate(data_loader):
    #     with torch.no_grad():
    #         seg_result = model(return_loss=False, rescale=True, **data)
    #         result = None
    #     results.append(result)
    #
    #     if verbose:
    #         vis_seg(data, seg_result, cfg.img_norm_cfg, data_id=i, colors=colors, score_thr=args.score_thr, save_dir=args.save_dir)
    #
    # batch_size = data['img'][0].size(0)
    #     for _ in range(batch_size):
    #         prog_bar.update()
    # return results

    # # original code end #


    # # new code begin: for testing and visualizing some imgs configured here
    #for i, data in enumerate(data_loader):

    panoList = ['GS__0004', 'GS__0007', 'GS__0015', 'GS__0016', 'GS__0017', 'GS__0018',
                'GS__0019', 'GS__0020', 'GS__0021', 'GS__0022', 'GS__0023', 'GS__0024',
                'GS__0025', 'GS__0026', 'GS__0027', 'GS__0028', 'GS__0029', 'GS__0030', 'GS__0031']
    iList = [4,7,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

    house = 'scene_02600'
    house = 'scene_03012'
    # house = 'scene_03060'
    # house = 'scene_03094'
    # house = 'scene_03142'
    # house = 'scene_03184'
    # house = 'scene_03199'
    imtype = 'equirect'
    imtype = ''
    # data/tmp/scene_02600/2D_rendering/4363/panorama/full/
    basedir = 'data/UP_lab_feb2021/test/'
    basedir = 'data/tmp' #/scene_02600/2D_rendering/4363/panorama/full/'
    #config = mmcv.Config.fromfile(config)
    #model.cfg = config
    #cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    i=0
    ind = 2

    panoList = listdir(basedir + imtype + '/' + house + '/2D_rendering/')

    for pano in panoList:
        #ind = ind + 1

        #pano = panoList[ind]
        print(pano)
        #i = iList[ind]

        img = basedir + imtype + '/' + pano + '.JPG'  # FTT mounted to undistorted_color_images
        img = basedir + imtype + '/' + house + '/2D_rendering/' + pano + '/panorama/full/rgb_warmlight.png'

        # check if img exists, if not, continue.
        if not path.exists(img):
            continue
        #cfg = model.cfg

        # prepare data
        data_tmp = dict(img=img)
        data_tmp = test_pipeline(data_tmp)

        data_tmp = scatter(collate([data_tmp], samples_per_gpu=1), [device])[0]
        # # Note WES: for test_ins_vis: for visualization, seg_result must be of type: list:1 > tuple:3 (3 tensor)
        seg_result_tmp = model(return_loss=False, rescale=True, **data_tmp)

        if verbose:
            #vis_seg(data_tmp, seg_result_tmp, cfg.img_norm_cfg, data_id=int(pano), colors=colors, score_thr=args.score_thr, save_dir=args.save_dir)
            vis_seg(data_tmp, seg_result_tmp, cfg.img_norm_cfg, data_id=int(pano), colors=colors,
                    score_thr=0.1, save_dir=args.save_dir)
    # # new code end #


    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--score_thr', type=float, default=0.25, help='score threshold for visualization')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--save_dir', help='dir for saveing visualized images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    assert not distributed
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args, cfg=cfg)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json(dataset, outputs, args.out)
                    coco_eval(result_files, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        coco_eval(result_files, eval_types, dataset.coco)

    # Save predictions in the COCO json format
    if args.json_out and rank == 0:
        if not isinstance(outputs[0], dict):
            results2json(dataset, outputs, args.json_out)
        else:
            for name in outputs[0]:
                outputs_ = [out[name] for out in outputs]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)


if __name__ == '__main__':
    main()
