# !/home/dig_ccm/anaconda/envs/train-scannet-solo-pt131-cu100/bin/python


# paths:
#   cfg:        /home/dig_ccm/projects/solov2_scannet/configs/solov2/
#   logfiles:   /home/dig_ccm/projects/solov2_scannet/work_dirs/solov2_scannet_extracted/
#   modelfile:  /home/dig_ccm/projects/solov2_scannet/work_dirs/solov2_test/
#                   oder
#               /home/dig_ccm/projects/solov2_scannet/work_dirs/solov2_scannet_extracted/
#                   oder
#               /home/dig_ccm/projects/atlantis/solov2_scannet-result/latest.pth
#   result img file:
#               /home/dig_ccm/projects/atlantis/solov2_scannet-result/

import sys

sys.path.insert(0, "../")
# wes
# from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result_pyplot

# config file for SOLO/SOLOv2 model for inference
config_file = '../configs/solo/solo_r50_fpn_1x_coco.py'
config_file = '../configs/solov2/solov2_r101_3x.py'

# checkpoint file; downloaded from github, SOLOv2 model was upgraded with tools/model_upgrade.py for upgrading mmdet 1.0 => mmdet >=2.0
checkpoint_file = '../checkpoints/SOLO_R50_1x.pth' # from WXinlong/SOLO homepage
checkpoint_file = '../checkpoints/SOLOv2_R101_3x_upgraded.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
# img = 'demo.jpg'

# basedir = '../data/matterportv1/'
basedir = '../data/UP_lab_feb2021/test/'
house = '1LXtFkjw3qL'  # scene name
# pano = '03a8325e3b054e3fad7e1e7091f9d283'
panoList = ['GS__0004', 'GS__0007', 'GS__0015', 'GS__0016', 'GS__0017', 'GS__0018',
            'GS__0019', 'GS__0020', 'GS__0021', 'GS__0022', 'GS__0023', 'GS__0024',
            'GS__0025', 'GS__0026', 'GS__0027', 'GS__0028', 'GS__0029', 'GS__0030', 'GS__0031']
#panoList = ['03a8325e3b054e3fad7e1e7091f9d283']

imtype = 'equirect'
imtype = ''
# imtype = 'mollweide0.00'
# imtype = 'mollweide0.50'
for pano in panoList:
    print(pano)
    # img = basedir + imtype + '/' + house + '/matterport_skybox_images/' +pano + '.jpg'
    img = basedir + imtype + '/' + pano + '.JPG'  # FTT mounted to undistorted_color_images

    result = inference_detector(model, img)

    #output_path = '../solov2-result/conf0_2/equirect-' + pano + '_fpn101_mmdet2_SOLOv2.png'
    output_path = '../solov2-result/conf0_1/' + pano + '_segmentation-result_mmdet2_SOLOv2.png'
    
    show_result_pyplot(img, result, output_path, model.CLASSES, 0.1)
   
