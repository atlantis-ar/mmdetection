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

# FTT changed to 101_3x
# config_file = '../configs/solov2/solov2_r101_3x.py' # => ftt seems to be SOLO
# config_file = '../configs/solov2/solov2_r101_3x_extracted.py' # => extracted from SOLOv2_R101_3x.pth
# config_file = '../configs/solov2/solov2_r50_1x_extracted.py' # => extracted from SOLOv2_R50_1x.pth
# config_file = '../configs/solov2/solo_r50_1x_extracted.py' # => extracted from SOLO_R50_1x.pth
# this config loads Coco database
# config_file = '../configs/solov2/solov2_extracted.py' # => ftt extracted from solov2.pth
# config_file = '../configs/solov2/solov2_r50_fpn_8gpu_1x.py'

# this config loads scanNet database
# config_file = '../configs/solov2/solov2_trainOnScannet.py' # => ftt extracted from solov2.pth

# WES: lightweight model file
# config_file = '../configs/solov2/SOLOv2_LIGHT_448_R50_3x.py' # => wes extracted from SOLOv2_LIGHT_448_R50_3x.pth

# config_file = '../configs/solov2/solo_r50_1x_extracted.py
# config_file = '../configs/solo/solo_r50_fpn_8gpu_1x.py'

#config_file = '../configs/solov2/solov2_extracted.py'
#config_file = '../configs/solov2/solov2_r50_fpn_8gpu_1x_extractedFromModel.py'
#config_file = '../configs/solov2/solo_r50_1x.py
config_file = '../configs/solo/solo_r50_fpn_1x_coco.py'
config_file = '../configs/solov2/solov2_r101_3x.py' #solov2_x101_dcn_fpn_8gpu_3x
#config_file = '../configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py'

# download the checkpoint from model zoo and put it in `checkpoints/`
# FTT
#checkpoint_file = '../checkpoints/solov2.pth' # from Google drive via Epiphqny/SOLOv2
# checkpoint_file = '../checkpoints/SOLOv2_R101_3x.pth' # from WXinlong/SOLO homepage
#checkpoint_file = '../checkpoints/SOLOv2_R50_1x.pth' # from WXinlong/SOLO homepage
#checkpoint_file = '../checkpoints/SOLOv2_R50_1x_upgraded.pth'  # from WXinlong/SOLO homepage
#checkpoint_file = '../checkpoints/solov2.pth'
checkpoint_file = '../checkpoints/SOLO_R50_1x.pth' # from WXinlong/SOLO homepage
checkpoint_file = '../checkpoints/SOLOv2_R101_3x_upgraded.pth'  #SOLOv2_X101_DCN_3x_upgraded.pth
#checkpoint_file = '../checkpoints/SOLOv2_X101_DCN_3x_upgraded.pth'

# WES # trainedwith val-tol2-scene0568_00.json
# checkpoint_file = '../work_dirs/solov2_test/val-tol2-scene0568_00_epoch1.pth'
# checkpoint_file = '../work_dirs/solov2_trainOnScannet/latest.pth'
# WES lightweight model files:
# checkpoint_file = '../checkpoints/SOLOv2_LIGHT_448_R50_3x.pth' # from WXinlong/SOLO homepage
# checkpoint_file = '../checkpoints/SOLO_R50_1x.pth' #converted by tools/upgrade_model.py
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R50_1x.pth' #converted by tools/upgrade_model.py

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
# img = 'demo.jpg'

# basedir = '/home/dig_ccm/PointGroup/data_mount/Matterport/v1/'
#basedir = '../data/matterportv1/'
basedir = '../data/UP_lab_feb2021/test/'
house = '1LXtFkjw3qL'  # scene name
# pano = '03a8325e3b054e3fad7e1e7091f9d283'
panoList = ['GS__0004', 'GS__0007', 'GS__0015', 'GS__0016', 'GS__0017', 'GS__0018',
            'GS__0019', 'GS__0020', 'GS__0021', 'GS__0022', 'GS__0023', 'GS__0024',
            'GS__0025', 'GS__0026', 'GS__0027', 'GS__0028', 'GS__0029', 'GS__0030', 'GS__0031']
#panoList = ['03a8325e3b054e3fad7e1e7091f9d283']
# pano = '0b302846f0994ec9851862b1d317d7f2'
# pano = '0f1ba9e425a0452eade2a180cfa41e32'
# pano = '0f7e0af0cb3b4c2abf62bba2fd955702'
# pano = '1e649cc84c9043b69e2367b7d5aeecf2'
# pano = '2b4fc2765e164775bb82e0aaf1d0d65d'
# pano = '126fdbcf213d4d5d95674ccba62fc72b'
imtype = 'equirect'
imtype = ''
# imtype = 'mollweide0.00'
# imtype = 'mollweide0.50'
for pano in panoList:
    print(pano)
    # img = basedir + imtype + '/' + house + '/matterport_skybox_images/' +pano + '.jpg'
    img = basedir + imtype + '/' + pano + '.JPG'  # FTT mounted to undistorted_color_images

    # l = dir(model)
    # d = model.__dict__
    #
    # print(l)
    # print(d)

    # for this, result should be type: tuple:2 (both list:80)
    result = inference_detector(model, img)

    # print(model.CLASSES) # type(tuple)

    # show the results, save it to path if not null
    # output_path = '~/projects/solov2/solov2/solov2-result'
    # output_path = '../solo-result/' + imtype + '-' + pano + '.png'
    # output_path = '../solo-result/conf0_0001/' + pano + '_segmentation-result_SOLO.png'
    # output_path = '../solo-result/conf0_3/' + pano + '_segmentation-result_SOLO.png'
    #output_path = '../solo-result/conf0_2/' + pano + '_segmentation-result_SOLO.png'

    #output_path = '../solov2-result/conf0_2/equirect-' + pano + '_fpn101_mmdet2_SOLOv2.png'
    output_path = '../solov2-result/conf0_1/' + pano + '_segmentation-result_mmdet2_SOLOv2.png'
    # model.show_result(img, result, score_thr=args.score_thr, wait_time=1, show=True)

    # show_result_pyplot(img, result, output_path, model.CLASSES, 0.15)

    show_result_pyplot(model, img, result, score_thr=0.1, out_file=output_path)   #, score_thr=0.1, fig_size=(15, 10))
    # show_result_pyplot(model, img, result, output_path, model.CLASSES, 0.1)

    ## these 2 lines were here in this file before
    ##show_result_pyplot(model, img, result, output_path, score_thr=0.15, fig_size=(15, 10))
    #show_result_pyplot(img, result, output_path, model.CLASSES, 0.1)

    # show_result(img, result, model.CLASSES, score_thr=0.3, wait_time=0, show=True, out_file=output_path)
    # save result img
    print("Finished")
