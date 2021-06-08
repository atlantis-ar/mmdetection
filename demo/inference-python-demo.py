import sys
from os import listdir
import os.path as osp
from os import path

from mmdet.apis import async_inference_detector, inference_detector, init_detector, show_result_pyplot

sys.path.insert(0, '../')

# config file for SOLO/SOLOv2 model for inference
# config_file = '../configs/solo/solo_r50_fpn_1x_coco.py'
config_file = '../configs/solov2/solov2_r101_3x.py'

# checkpoint file; downloaded from github, SOLOv2 model was upgraded with
#     tools/model_upgrade.py for upgrading mmdet 1.0 => mmdet >=2.0
#     from WXinlong/SOLO homepage
# checkpoint_file = '../checkpoints/SOLO_R50_1x.pth'
checkpoint_file = '../checkpoints/SOLOv2_R101_3x_upgraded.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
imgName = 'demo.jpg'
basedir = '../data/test/'
img = basedir + imgName + '.JPG'

result = inference_detector(model, img)

output_path = '../solov2-result/' + imgName + \
              '_segmentation-result_mmdet2_SOLOv2.png'

show_result_pyplot(model, img, result, score_thr=0.1, out_file=output_path)
