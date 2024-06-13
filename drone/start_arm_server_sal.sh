#! /bin/bash



#! /bin/bash

# necessary if used for the Allied Vision drivers
export GENICAM_GENTL64_PATH="/opt/Vimba_v4.2_ARM64/Vimba_4_2/VimbaUSBTL/CTI/arm_64bit"

# add efficientdet and utils to the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:"detection/efficientdet"
export PYTHONPATH=$PYTHONPATH:"../"

python3 arm_server.py --detector trt_checkpoints/resnet18_960x540_trt8.trt --image_size 1280x720 --detector_size 960x540 --detection_frequency 1 --main_image_scale 0.5 --cam_type demo --saliency

