################################################################### Test ######################################################################
./tools/dist_test.sh configs/top_down/resnet/coco/res50_coco_256x192.py checkpoint/res50_coco_256x192-ec54d7f3_20200709.pth 1 --eval mAP
./tools/dist_test.sh configs/top_down/darkpose/coco/hrnet_w32_coco_256x192_dark.py checkpoint/hrnet_w32_coco_256x192_dark-07f147eb_20200812.pth 1 --eval mAP
./tools/dist_test.sh configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py checkpoint/hrnet_w32_coco_256x192-c78dce93_20200708.pth 1 --eval mAP
./tools/dist_test.sh configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py checkpoint/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth 1 --eval mAP

################################################################### Train ######################################################################
./tools/dist_train.sh configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py 1
