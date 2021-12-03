#!/bin/bash

#FTT Use sudo and your Namenskürzel as param1 or "unmount" to unmount all dirs
#otherwise
#map certain Matterport v1 subdirs for inference to <current_dir>/data/matterportv1


if [ "$1" = "unmount" ]; then
 echo "---unmounting all mounted windows dirs---"
 umount data/scannet/annotations
 umount data/scannet/test
 umount data/scannet/train
 umount data/scannet/val
 exit 0
fi


echo "---creating scannet dirs---"
mkdir data/scannet

echo "---mounting scannet windows dirs---"
target=annotations
mkdir data/scannet/$target
mount -t cifs -o user=$1,domain=jr1 //143.224.133.11/Storage/datasets/ScanNet/solov2/$target/ data/scannet/$target

target=test
mkdir data/scannet/$target
mount -t cifs -o user=$1,domain=jr1 //143.224.133.11/Storage/datasets/ScanNet/solov2/$target/ data/scannet/$target

target=train
mkdir data/scannet/$target
mount -t cifs -o user=$1,domain=jr1 //143.224.133.11/Storage/datasets/ScanNet/solov2/$target/ data/scannet/$target

target=val
mkdir data/scannet/$target
mount -t cifs -o user=$1,domain=jr1 //143.224.133.11/Storage/datasets/ScanNet/solov2/$target/ data/scannet/$target




