################################################################### Test ######################################################################
# simplebaseline resnet50
./tools/dist_test.sh configs/top_down/resnet/coco/res50_coco_256x192.py checkpoint/res50_coco_256x192-ec54d7f3_20200709.pth 1 --eval mAP
# Darknet HRNet
./tools/dist_test.sh configs/top_down/darkpose/coco/hrnet_w32_coco_256x192_dark.py checkpoint/hrnet_w32_coco_256x192_dark-07f147eb_20200812.pth 1 --eval mAP
# HRNet
./tools/dist_test.sh configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py checkpoint/hrnet_w32_coco_256x192-c78dce93_20200708.pth 1 --eval mAP
./tools/dist_test.sh configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py checkpoint/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth 1 --eval mAP

PORT=29501 ./tools/dist_test.sh configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py work_dirs/hrnet_w32_coco_256x192/best.pth 2 --eval mAP

# Lite-HRNet
./tools/dist_test.sh configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py checkpoint/litehrnet_18_coco_256x192.pth 1 --eval mAP
./tools/dist_test.sh configs/top_down/lite_hrnet/coco/litehrnet_30_coco_256x192.py checkpoint/litehrnet_30_coco_256x192.pth 1 --eval mAP
./tools/dist_test.sh configs/top_down/naive_litehrnet/coco/naive_litehrnet_18_coco_256x192.py checkpoint/naive_litehrnet_18_coco_256x192.pth 1 --eval mAP
./tools/dist_test.sh configs/top_down/naive_litehrnet/coco/wider_naive_litehrnet_18_coco_256x192.py checkpoint/wider_naive_litehrnet_18_coco_256x192.pth 1 --eval mAP

./tools/dist_test.sh configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py work_dirs/litehrnet_18_coco_256x192/best.pth 1 --eval mAP
################################################################### Train ######################################################################
# HRNet
./tools/dist_train.sh configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py 1 --resume-from work_dirs/hrnet_w32_coco_256x192/epoch_10.pth

# Lite-HRNet
./tools/dist_train.sh configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py 4 --autoscale-lr  # --gpu-ids 4,5,6,7
./tools/dist_train.sh configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py 4 --autoscale-lr --resume-from work_dirs/litehrnet_18_coco_256x192/epoch_210.pth

./tools/dist_train.sh configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192_dark.py 1 --autoscale-lr 
./tools/dist_train.sh configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192_dark.py 2 --autoscale-lr  --resume-from work_dirs/litehrnet_18_coco_256x192_dark/epoch_60.pth

#EAHRnet
./tools/dist_train.sh configs/top_down/eahrnet/coco/eahrnet_18_coco_256x192.py 4 --autoscale-lr